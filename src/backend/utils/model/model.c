#include <sys/stat.h>
#include <errno.h>
#include <math.h>
#include <curl/curl.h>

#include "postgres.h"
#include "c.h"
#include "fmgr.h"
#include "miscadmin.h"


#include "access/genam.h"
#include "access/heapam.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/stratnum.h"
#include "catalog/indexing.h"
#include "catalog/pg_attribute.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "utils/lsyscache.h"
#include "executor/tuptable.h"
#include "executor/spi.h"
#include "utils/fmgroids.h"
#include "utils/snapmgr.h"


#include "access/attnum.h"
#include "access/tupdesc.h"
#include "catalog/pg_type.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "nodes/parsenodes.h"
#include "tcop/dest.h"
#include "utils/builtins.h"
#include "utils/c_api.h"
#include "utils/errcodes.h"
#include "utils/jsonb.h"
#include "utils/model.h"
#include "utils/rel.h"
#include "utils/syscache.h"


#define QUOTEMARK '"'
#define URL_REQUEST "http://localhost:8000/ml"

#define SPI_natts SPI_tuptable->tupdesc->natts

typedef bool (*GetFeatureIndicesFunction)(ModelCalcerHandle* modelHandle, size_t** indices, size_t* count);

static char** GetFeatures(ModelCalcerHandle* modelHandle, GetFeatureIndicesFunction getFeaturesFun, char** featureNames, size_t *count);
static char** getModelFeatures(ModelCalcerHandle *modelHandle, size_t *featureCount);

static char * numeric_to_cstring(Numeric n);
static void modelTypeToChar(ModelType modelType, char model_type[2]);

static TupleDesc GetMlModelTableDesc(void);
static char** CreateArrayFromList(List *cols);
static ModelCalcerHandle * GetMlModelByName(const char * name, char **classes_json_str, char **loss_function, char* model_out_type);
static char * GetFeaturesInfo(ModelCalcerHandle *modelHandle, int *resultLen);
static char* CreateJsonModelParameters(CreateModelStmt *stmt);
static const char* TransformMetric(char* metric);

static Datum LoadFileToBuffer(const char * tmp_name,  int file_length,void **model_buffer);
static Datum GetFeaturesFieldInfo(void* model_buffer, int file_length, char **parms, char** classes);
static void CreateTemplateTypesOfRecord(ModelCalcerHandle *modelHandle, TupleDesc tupdesc, int32** arrTypes);
static void SetPredictionToModel(char* loss_function, ModelCalcerHandle **modelHandle);
static void CreatePredictInputData(TupleDesc tupdesc, int32 count, int32 *arrTypes, Datum *values, float4 **arrFloat, char*** arrCat);

static char * GetLossFunctionFromParms(char* parms);
static char ** GetClassesFromJson(char* classes_json_str);
static char * ArrayToStringList(char **featureName, int featureCount);
static char * IntArrayToStringList(size_t *featureData, int featureCount);
static void SaveSidToMetadata( uint sid, char *modelname, ModelType modelType, char *str_parameter, char *queryString);
static void SendRequest(int32 sid);
static int LoadModelFromFileAndSaveToMetadata(const char* tmp_name, char* modelname, ModelType model_type,
												Datum acc, char* str_parameter);

static bool checkInArray(char* name, char **features, int32 featureCount, int32 *retIndex);

static Form_pg_class GetPredictTableFormByName(const char *tablename);
static char * read_whole_file(const char *filename, int *length);
static double sigmoid(double x);
static bool pstrcasecmp(char  *s1, char  *s2);

// сделать shmem
static Oid mlJsonWrapperOid = InvalidOid;
// static Oid PredictTableOid = InvalidOid;
static Oid MetadataTableOid = InvalidOid;
static Oid MetadataTableIdxOid = InvalidOid;

inline static double
sigmoid(double x) {
	return 1. / (1. + exp(-x));
}


static char *
numeric_to_cstring(Numeric n)
{
	Datum		d = NumericGetDatum(n);

	return DatumGetCString(DirectFunctionCall1(numeric_out, d));
}


/*
* case compare column name and model feature name
* and replace symbol '-' to '_'
* as the postgers can't use symbol '-' in column name
*/
static bool
pstrcasecmp(char  *s1, char  *s2)
{
	char *p1,*p2, pp1, pp2;
	p1=s1;
	p2=s2;

	while (*p1 && *p2)
	{
		if (isalpha(*p1))
			pp1 = tolower(*p1);
		else
			if (*p1 == '-')
				pp1 = '_';
			else
				pp1 = *p1;

		if (isalpha(*p2))
			pp2 = tolower(*p2);
		else
			if (*p2 == '-')
				pp2 = '_';
			else
				pp2 = *p2;

		if (pp1 == '_' && strcasecmp(p1+1,"id") == 0 && strcasecmp(p2,"ID") == 0)
			return true;

		if (pp2 == '_' && strcasecmp(p2+1,"id") == 0 && strcasecmp(p1,"ID") == 0)
			return true;

		if (pp1!= pp2)
		{
			return false;
		}

		p1 ++;
		p2 ++;
	}

	if (*p1 !=*p2)
		return false;

	return true;
}


static bool
checkInArray(char* name, char **features, int32 featureCount, int32 *retIndex)
{
	int i;
	for(i=0; i < featureCount; i++)
	{
		if (!features[i])
			elog(ERROR, "the feature %d is NULL", i);
		if ( pstrcasecmp(features[i], name) ){
			*retIndex = i;
			return true;
		}
	}
	*retIndex = -1;
	return false;
}


/*
 * Get a tuple descriptor for CREATE MODEL
 */
TupleDesc
GetCreateModelResultDesc(void)
{
	TupleDesc   tupdesc;

	/* need a tuple descriptor representing three TEXT columns */
	tupdesc = CreateTemplateTupleDesc(1);
	TupleDescInitEntry(tupdesc, (AttrNumber) 1, "accuracy",
	   TEXTOID, -1, 0);
	return tupdesc;
}

/*
 * Create string array output of columns from PredictModelStmt->cols 
 */
static char**
CreateArrayFromList(List *cols)
{
	int32 col_count = list_length(cols);
	char **out = palloc0(sizeof(char*) * col_count);
	int32 i = 0;
	ListCell *lc;

	foreach(lc, cols)
	{
		String *str;
		str = (String *) lfirst(lc);
		out[i] = pstrdup(str->sval);
		i++;
	}
	return out;
}


/*
 * Get a tuple descriptor for ml+nodel table
 */
static TupleDesc
GetMlModelTableDesc(void)
{

	TupleDesc   tupdesc = CreateTemplateTupleDesc(Natts_model);

	TupleDescInitEntry(tupdesc, 1, "name", NAMEOID, -1, 0);
	TupleDescInitEntry(tupdesc, 2, "fieldlist", TEXTOID, -1, 0);
	TupleDescInitEntry(tupdesc, 3, "model_type", BPCHAROID, -1, 0); // 1042
	TupleDescInitEntry(tupdesc, 4, "acc", FLOAT4OID, -1, 0);
	TupleDescInitEntry(tupdesc, 5, "info", TEXTOID, -1, 0);
	TupleDescInitEntry(tupdesc, 6, "args", TEXTOID, -1, 0);
	TupleDescInitEntry(tupdesc, 7, "data", BYTEAOID, -1, 0);
	TupleDescInitEntry(tupdesc, 8, "classes", TEXTOID, -1, 0);
	TupleDescInitEntry(tupdesc, 9, "loss_function", TEXTOID, -1, 0);
	TupleDescInitEntry(tupdesc, 10, "sid", INT4OID, -1, 0);
	TupleDescInitEntry(tupdesc, 11, "query", TEXTOID, -1, 0);
	return tupdesc;
}

static void
modelTypeToChar(ModelType modelType, char model_type[2])
{
	
	switch (modelType)
	{
		case MODEL_TYPE_REGRESSION:
			model_type[0] = 'R';
			break;
		case MODEL_TYPE_RANKING:
			model_type[0] = 'G';
			break;
		case MODEL_TYPE_CLASSIFICATION:
			model_type[0] = 'C';
			break;
		case MODEL_TYPE_UNDEFINED:
			model_type[0] = 'U';
			break;
		default:
			elog(ERROR, "undefined model type");
	}

	return (char*) model_type;
}



/* TODO:  внедрить следующие метрики:
'Logloss', 'CrossEntropy', 'CtrFactor', 'Focal', 'RMSE', 'LogCosh', 'Lq', 'MAE', 'Quantile', 'MultiQuantile', 'Expectile',
'LogLinQuantile', 'MAPE', 'Poisson', 'MSLE', 'MedianAbsoluteError', 'SMAPE', 'Huber', 'Tweedie', 'Cox', 'RMSEWithUncertainty',
'MultiClass', 'MultiClassOneVsAll', 'PairLogit', 'PairLogitPairwise', 'YetiRank', 'YetiRankPairwise', 'QueryRMSE',
'QuerySoftMax', 'QueryCrossEntropy', 'StochasticFilter', 'LambdaMart', 'StochasticRank', 'PythonUserDefinedPerObject',
'PythonUserDefinedMultiTarget', 'UserPerObjMetric', 'UserQuerywiseMetric', 'R2', 'NumErrors', 'FairLoss', 'AUC', 'Accuracy',
'BalancedAccuracy', 'BalancedErrorRate', 'BrierScore', 'Precision', 'Recall', 'F1', 'TotalF1', 'F', 'MCC', 'ZeroOneLoss', 'HammingLoss',
'HingeLoss', 'Kappa', 'WKappa', 'LogLikelihoodOfPrediction', 'NormalizedGini', 'PRAUC', 'PairAccuracy', 'AverageGain', 'QueryAverage',
'QueryAUC', 'PFound', 'PrecisionAt', 'RecallAt', 'MAP', 'NDCG', 'DCG', 'FilteredDCG', 'MRR', 'ERR', 'SurvivalAft', 'MultiRMSE',
'MultiRMSEWithMissingValues', 'MultiLogloss', 'MultiCrossEntropy', 'Combination'.
*/
static const char*
TransformMetric(char* metric)
{
	if (strcmp(metric, "logloss") == 0)
	{
		return "Logloss";
	}

	if (strcmp(metric, "multiclass") == 0)
	{
		return "MultiClass";
	}

	if (strcmp(metric, "auc") == 0)
	{
		return "AUC";
	}
	if (strcmp(metric, "ndcg") == 0)
	{
		return "NDCG";
	}

	return "";
}

/* create in model options parameter*/
static char*
CreateJsonModelParameters(CreateModelStmt *stmt)
{
	int len;
	StringInfoData  buf;
	ListCell  *lc;
	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');

	foreach(lc, stmt->options)
	{
		ModelOptElement *opt;
		opt = (ModelOptElement *) lfirst(lc);

		switch(opt->parm)
		{
			case MODEL_PARAMETER_TARGET:
				appendStringInfo(&buf, "\"target\":\"%s\"", (char*)opt->value);
				break;

			case MODEL_PARAMETER_EVAL_METRIC:
				appendStringInfo(&buf, "\"eval_metric\":\"%s\"", TransformMetric((char*)opt->value));
				break;

			case MODEL_PARAMETER_LOSS_FUNCTION:
				if (strcmp(opt->value, "logloss") == 0)
				{
					appendStringInfo(&buf, "\"loss_function\":\"Logloss\"");
					break;
				}
				if (strcmp(opt->value, "crossentropy") == 0)
				{
					appendStringInfo(&buf, "\"loss_function\":\"CrossEntropy\"");
					break;
				}
				if (strcmp(opt->value, "yetirank") == 0)
				{
					appendStringInfo(&buf, "\"loss_function\":\"YetiRank\"");
					break;
				}
				if (strcmp(opt->value, "querysoftmax") == 0)
				{
					appendStringInfo(&buf, "\"loss_function\":\"QuerySoftMax\"");
					break;
				}
				if (strcmp(opt->value, "multiclass") == 0)
				{
					appendStringInfo(&buf, "\"loss_function\":\"MultiClass\"");
					break;
				}
				break;

			case MODEL_PARAMETER_IGNORE:
				if (opt->value)
				{
					appendStringInfo(&buf, "\"ignored\":[\"%s\"]",opt->value);
				}
				else
				{
					ListCell  *lc2;
					StrModelElement *el;
					appendStringInfo(&buf, "\"ignored\":[");
					foreach( lc2, opt->elements)
					{
						el = (StrModelElement *) lfirst(lc2);
						appendStringInfo(&buf,"\"%s\",", el->value);
					}
					len = buf.len;
					*(buf.data + len - 1) = ']';
				}
				break;
			case MODEL_PARAMETER_GROUP_BY:
				appendStringInfo(&buf, "\"group_by\":\"%s\"",opt->value);
				break;

			default:
elog(ERROR, "params is undefined num=%d", opt->parm);
		}

		appendStringInfo(&buf, ",");
	}

	len = buf.len;
	*(buf.data + len - 1) = '}';
	return buf.data;
}


static Datum
LoadFileToBuffer(const char * tmp_name,  int file_length, void **model_buffer)
{
	bytea *result;
	int len;
	result = (text *) palloc(file_length + VARHDRSZ);
	*model_buffer = read_whole_file(tmp_name, &len);

	SET_VARSIZE(result, file_length + VARHDRSZ);
	memcpy(VARDATA(result), *model_buffer, file_length);
	
	return PointerGetDatum(result);
}

/*
 * deallocate the result of function
 */
static Datum 
GetFeaturesFieldInfo(void* model_buffer, int file_length, char** parms, char** classes)
{
	char *modelInfo, *info, *classes_tmp;
	int len;
	text *infoOutDatum;
	ModelCalcerHandle *modelHandle = ModelCalcerCreate();
	if (!LoadFullModelFromBuffer(modelHandle, model_buffer, file_length))
	{
		elog(ERROR, "LoadFullModelFromBuffer error message: %s\n", GetErrorString());
	}

	info = (char*) GetModelInfoValue(modelHandle, "params", 6); // strlen("parms")
	modelInfo = GetFeaturesInfo(modelHandle, &len);

	classes_tmp = (char*) GetModelInfoValue(modelHandle, "class_params", 12); // strlen("class_parms")
	if (classes_tmp)
	{
		*classes = pstrdup(classes_tmp);
	}

	infoOutDatum = (text *) palloc(len + VARHDRSZ);
	SET_VARSIZE(infoOutDatum, len + VARHDRSZ);
	memcpy(VARDATA(infoOutDatum), modelInfo, len);
	*parms = pstrdup(info);

	// free(parms) ???
	// free(info); // ????
	ModelCalcerDelete(modelHandle);

	return PointerGetDatum(infoOutDatum);
}


static char *
GetFeaturesInfo(ModelCalcerHandle *modelHandle, int *resultLen)
{
	char *strbuf, *data;
	size_t *indices;
	size_t featureCount;
	char** featureName ; 

	StringInfo outInfoString = makeStringInfo();

	MemoryContext resultcxt, oldcxt;

	/* This is the context that we will allocate our output data in */
	resultcxt =  AllocSetContextCreate(TopMemoryContext,
			"FeatureInfoContext",
			ALLOCSET_DEFAULT_SIZES);

	oldcxt = MemoryContextSwitchTo(resultcxt);


	featureName = palloc(sizeof(void*) * FIELDCOUNT);
	if (! GetModelUsedFeaturesNames(modelHandle, &featureName, &featureCount))
	{
		elog(ERROR,"get model feature names error: %s", GetErrorString());
	}

	if (featureCount > FIELDCOUNT)
		elog(ERROR, "count of field %ld is overflow.", featureCount);


	strbuf = ArrayToStringList(featureName, featureCount);

	appendStringInfo(outInfoString, "{ \"fieldList\":\"%s\",", strbuf);
	
		
	if (!GetCatFeatureIndices(modelHandle, &indices, &featureCount))
	{
		elog(ERROR,"CatBoost error: %s", GetErrorString());
	}


	strbuf = IntArrayToStringList(indices, featureCount);
	appendStringInfo(outInfoString, " \"CategoryFieldList\":\"%s\",", strbuf);
	free(indices);


	if (!GetFloatFeatureIndices(modelHandle, &indices, &featureCount))
	{
		elog(ERROR,"CatBoost error: %s", GetErrorString());
	}


	strbuf = IntArrayToStringList(indices, featureCount);
	appendStringInfo(outInfoString, " \"FloatFieldList\":\"%s\"}", strbuf);
	// pfree(strbuf);
	free(indices);


	data = outInfoString->data;

	MemoryContextSwitchTo(oldcxt);
	MemoryContextReset(resultcxt);
	
	// pfree(outInfoString);
	*resultLen = outInfoString->len;
	return data;
}


static char *
ArrayToStringList(char **featureName, int featureCount)
{
	int i = 0;
	char *p;
	char * strbuf = palloc( NAMEDATALEN * featureCount);
	
	p = strbuf;

	for(i = 0; i < featureCount; i++)
	{
		if (featureName[i] == NULL)
			elog(ERROR, "feature #%d is null", i);

		strcpy(p, featureName[i]);
		p += strlen(featureName[i]);
		*p = ',';
		p++;
	}
	
	*--p = '\0';
	return strbuf;
}


static char *
IntArrayToStringList(size_t *featureData, int featureCount)
{
	int i = 0;
	char * data;
	StringInfo buf = makeStringInfo();

	for(i = 0; i < featureCount; i++)
	{
		appendStringInfo(buf, "%ld,", featureData[i]);
	}

	*(buf->data + buf->len -1) = '\0';
	data = 	buf->data;
	pfree(buf);
	return data;
}


/**
 * @TODO тут нужно распарсить tablename, если в нем указано tablespace, то использовать tablespace
 * и еще посмотреть в path_search.
 * */
static Form_pg_class
GetPredictTableFormByName(const char *tablename)
{
	HeapTuple tup;
	Form_pg_class form;
	Oid PredictTableOid;

	PredictTableOid = get_relname_relid(tablename,(Oid) PG_PUBLIC_NAMESPACE);

	if (!PredictTableOid)
		elog(ERROR, "tablename %s not found in public", tablename);

	tup = SearchSysCache1(RELOID, ObjectIdGetDatum(PredictTableOid));

	if (!HeapTupleIsValid(tup))
		elog(ERROR, "cache lookup failed for relation %d", PredictTableOid);
	form = (Form_pg_class) GETSTRUCT(tup);

	ReleaseSysCache(tup);
	return form;
}



/*
 * Get a tuple descriptor for PREDICT MODEL
 */

TupleDesc GetPredictModelResultDesc(PredictModelStmt *node){

	TupleDesc   tupdesc;
	ListCell *lc;
	int32 i = 1;
	int32 col_count = list_length(node->cols);


	tupdesc = CreateTemplateTupleDesc(col_count + 1);

	foreach(lc, node->cols)
	{
		String *str;
		str = (String *) lfirst(lc);
	
		TupleDescInitEntry(tupdesc, (AttrNumber) i,
			str->sval, TEXTOID, -1, 0);
		i++;
	}

	TupleDescInitEntry(tupdesc, (AttrNumber) i, "ml_result",
			TEXTOID, -1, 0);

	return tupdesc;

}


static void
CreateTemplateTypesOfRecord(ModelCalcerHandle *modelHandle, TupleDesc tupdesc, int32** arrTypes)
{
	int32 i,j;
	size_t *  arrFloat;
	size_t *  arrCat;
	char **featureNames;
	size_t  cat_count, float_count, featureCount;
	int32 table_natts = tupdesc->natts;
	int32 *p = *arrTypes;
	int32 *model2Table;


	if (!GetModelUsedFeaturesNames(modelHandle, &featureNames, &featureCount))
	{
		elog(ERROR,"get model feature names: %s", GetErrorString());
	}

	model2Table = palloc0(sizeof(int32) * featureCount);
	memset(p, -1, sizeof(int32) * table_natts);

	for (i = 0; i < featureCount; i++)
	{	
		// p[i] = -1;
		for (j=0; j < table_natts; j++)
		{
			if (strcmp(tupdesc->attrs[j].attname.data, featureNames[i]) == 0)
			{
				model2Table[i] = j;
				break;
			}
		}
	}

	if (!GetFloatFeatureIndices(modelHandle, &arrFloat , &float_count))
	{
		elog(ERROR,"get model float feature indexes: %s", GetErrorString());
	}

	if (!GetCatFeatureIndices(modelHandle, &arrCat , &cat_count))
	{
		elog(ERROR,"get model categorical feature indexes: %s", GetErrorString());
	}

	if (arrFloat)
	{
		j = 0;
		for (i=0; i < featureCount; i++)
		{
			if (i == arrFloat[j])
			{
				p[model2Table[i]] = 1000 + j;
				j++;
				continue;
			}

			if (j > float_count)
				elog(ERROR, "feature count is owerflow");
		}
	}


	if (arrCat)
	{
		j=0;
		for (i=0; i < featureCount; i++)
		{
			if (i == arrCat[j])
			{
				p[model2Table[i]] = j;
				j++;
				continue;
			}
			if (j > cat_count)
				elog(ERROR, "feature count is owerflow");
		}
	}

	pfree(model2Table);
	free(arrFloat); // allocated in c_api GetFloatFeatureIndices
	free(arrCat);   // allocated in c_api GetCatFeatureIndices
	free(featureNames); // возможно стоит удалить  каждый элемент featureNames
}

static void
SetPredictionToModel(char* loss_function, ModelCalcerHandle **modelHandle)
{
	if (strcmp(loss_function,"Logloss") == 0)
	{
		// modelclass = MODEL_TYPE_CLASSIFICATION;
		// elog(WARNING,"loss %s APT_RAW_FORMULA_VAL", loss_function);
		if (!SetPredictionType(*modelHandle, APT_RAW_FORMULA_VAL))
		{
			elog(ERROR, "prediction type error %s", GetErrorString());
		}
	}
	else if (strcmp(loss_function,"MultiClass") == 0)
	{
		// modelclass = MODEL_TYPE_CLASSIFICATION;
		// elog(WARNING,"loss %s APT_CLASS", loss_function);
		if (!SetPredictionType(*modelHandle, APT_CLASS))
		{
			elog(ERROR, "prediction type error %s", GetErrorString());
		}
	}
	else
	{
		// elog(WARNING,"loss %s APT_RAW_FORMULA_VAL", loss_function);
		// modelclass = MODEL_TYPE_REGRESSION;
		if (!SetPredictionType(*modelHandle, APT_RAW_FORMULA_VAL))
		{
			elog(ERROR, "prediction type error %s", GetErrorString());
		}
	}

}


static void
CreatePredictInputData(TupleDesc tupdesc, int32 count, int32* arrTypes, Datum *values, float4 **arrFloatOut, char ***arrCatOut)
{
	int32 float_id, cat_id, i;
	float4 *arrFloat;
	char **arrCat;
	arrFloat = *arrFloatOut;
	arrCat = *arrCatOut;

	for (i=0; i < count; i++)
	{
		if (arrTypes[i] >= 1000)
		{
			float_id = arrTypes[i] - 1000;
			switch (tupdesc->attrs[i].atttypid)
			{
		case FLOAT4OID:
			arrFloat[float_id] = (float)DatumGetFloat4(values[i]);
			break;
		case FLOAT8OID:
			arrFloat[float_id] = (float)DatumGetFloat8(values[i]);
			break;
		case INT4OID:
			arrFloat[float_id] = (float) DatumGetInt32(values[i]);
			break;
		case INT8OID:
			arrFloat[float_id] = (float)DatumGetInt64(values[i]);
			break;
		case INT2OID:
			arrFloat[float_id] = (float)DatumGetInt16(values[i]);
			break;
		case BOOLOID:
			arrFloat[float_id] = (float)DatumGetBool(values[i]);
			break;
		default:
			elog(ERROR,"num field[%d] %s type oid=%d undefined", i, NameStr(tupdesc->attrs[i].attname), tupdesc->attrs[i].atttypid);
					}
		}

		if (arrTypes[i] >= 0 && arrTypes[i] < 1000)
		{
			cat_id = arrTypes[i];
			switch (tupdesc->attrs[i].atttypid)
			{
case TEXTOID:
case BPCHAROID:
	arrCat[cat_id] = (char*) TextDatumGetCString(values[i]);
	break;
default:
	elog(ERROR,"cat field[%d] %s type oid=%d undefined", i, NameStr(tupdesc->attrs[i].attname), tupdesc->attrs[i].atttypid);
			}
		}
	}
}


static char**
GetFeatures(ModelCalcerHandle* modelHandle, GetFeatureIndicesFunction getFeaturesFun, char** featureNames, size_t *count)
{
	size_t *indices = NULL, indices_cnt;
	char **modelFeatureNames = NULL;

	if (getFeaturesFun(modelHandle, &indices, &indices_cnt))
	{
		int32 i;
		modelFeatureNames = palloc( sizeof(char*) * indices_cnt);

		for (i=0; i < indices_cnt; i++)
		{
			modelFeatureNames[i] = pstrdup(featureNames[indices[i]]);
		}

		free(indices); // allocated by CatBoostModel::Get_Xxx_FeatureIndices()
	} 
	else
	{
		elog(ERROR,"error in GetFloatFeatureIndices() %s", GetErrorString());
	}

	*count = indices_cnt;
	return modelFeatureNames;
}


/*
*  get array names Model features
*/
static char**
getModelFeatures(ModelCalcerHandle *modelHandle, size_t *featureCount)
{
	char** featureName = palloc(FEATURES_BUFSIZE);

	if (!GetModelUsedFeaturesNames(modelHandle, &featureName, featureCount))
	{
		elog(ERROR,"get model feature name error: %s", GetErrorString());
	}
	return featureName;
}


void
PredictModelExecuteStmt(PredictModelStmt *stmt, const char *queryString, DestReceiver *dest)
{
	Relation rel;
	HeapTuple tup;
	TableScanDesc scan;
	SysScanDesc sscan;
	TupleDesc tupdesc;
	TupOutputState *tstate;
	Datum *values, *outvalues;
	bool *nulls, *outnulls;
	ScanKeyData skey[1];
	Form_pg_class form;
	Oid PredictTableOid;
	ModelCalcerHandle *modelHandle;
	int32  table_natts, i, len, ret;
	int32 *  arrTypes;
	char **arrCatNames = NULL, *classes_json_str = NULL, **classes = NULL;
	char **arrFloatNames = NULL;  // массив имен

	size_t model_dimension;
	size_t *float_indices, *cat_indices;
	size_t float_indices_cnt, cat_indices_cnt, featureCount;
	char **featureNames;

	int32 cat_feature_counter = 0, float_feature_counter = 0;
	int32 *iscategory, *feature_idx, idx, *out_names;

	char** arrCat = NULL;
	double *arrFloat = NULL;

	double* result_pa;
	size_t cat_cnt, float_cnt;	
	// bool isFound = false;
	char *loss_function;
	// ModelType modelclass;
	int32 j=0, max_probability_idx, rows;
	float4 max_probability;
	char model_type;
	SPITupleTable *tuptable;
	MemoryContext oldCtx, resultcxt;

	StringInfoData query;
	char **arrOutNames = NULL;

	SelectStmt *select = (SelectStmt *) stmt->query;
	
	List *targetList = select->targetList;
	Assert( targetList != NULL);

	ListCell *cell = list_head(targetList);
	ResTarget *target = lfirst_node(ResTarget, cell);

	/* This is the context that we will allocate our output data in */
	resultcxt =  AllocSetContextCreate(TopMemoryContext,
			"FeatureDataContext",
			ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(resultcxt);

	initStringInfo(&query);
	appendStringInfo(&query, "SELECT %s", queryString + target->location);

	// get model
	modelHandle = GetMlModelByName((const char*)stmt->modelname, &classes_json_str, &loss_function, &model_type);

	// loss_function = pstrdup("Logloss");
	// SetPredictionToModel(loss_function, &modelHandle);

	model_dimension = (size_t)GetDimensionsCount(modelHandle);
	result_pa  = (double*) palloc( sizeof(double) * model_dimension);

	featureNames = getModelFeatures(modelHandle, &featureCount);

	//  получаем имена полей
	arrCatNames = GetFeatures(modelHandle, GetCatFeatureIndices, featureNames, &cat_indices_cnt);
	arrFloatNames = GetFeatures(modelHandle, GetFloatFeatureIndices, featureNames, &float_indices_cnt);


	// выходной дескриптор
	tupdesc = GetPredictModelResultDesc(stmt);

	arrOutNames = CreateArrayFromList(stmt->cols);


	len = list_length(stmt->cols) + 1;

	tstate = begin_tup_output_tupdesc(dest, tupdesc, &TTSOpsVirtual);

	outnulls = palloc( sizeof(bool) * len);
	outvalues = palloc( sizeof(Datum) * len);


	SPI_connect();

	ret = SPI_exec(query.data, 2);
	rows = SPI_processed;
	tuptable = SPI_tuptable;
	if (ret < 1 || tuptable == NULL)
	{
		elog(ERROR, "Query errorcode=%d", ret);
	}


	iscategory = (int32*)palloc0(SPI_natts * sizeof(int32));
	feature_idx = (int32*)palloc0(SPI_natts * sizeof(int32));
	out_names = (int32*)palloc0(SPI_natts * sizeof(int32));

	for(i=0; i < SPI_natts; i++)
	{
		if ( checkInArray(ModelGetFieldName(i), arrOutNames, len-1, &idx))
		{
			out_names[i] = idx;
		}
		else
			out_names[i] = -1;

		if(checkInArray(ModelGetFieldName(i), arrCatNames, cat_indices_cnt, &idx))
		{
			iscategory[i] = ML_FEATURE_CATEGORICAL;
			feature_idx[i] = idx;
			cat_feature_counter ++;
			continue;
		}
		if ( checkInArray(ModelGetFieldName(i), arrFloatNames, float_indices_cnt, &idx))
		{
			float_feature_counter ++;
			iscategory[i] = ML_FEATURE_FLOAT;
			feature_idx[i] = idx;
			continue;
		}

		iscategory[i] = ML_FEATURE_NONE;
		feature_idx[i] = -1;

	}

	//  check model features
	if (float_feature_counter != float_indices_cnt)
	{
		elog(ERROR,
			"count of numeric features is not valid, must be %ld is %ld",
			float_indices_cnt, float_feature_counter
		);
	}

	if (cat_feature_counter != cat_indices_cnt)
	{
		elog(ERROR,
			"count of categocical features is not valid, must be %ld is %ld",
			cat_indices_cnt, cat_feature_counter
		);
	}


	arrCat = palloc0(sizeof(char*) * cat_indices_cnt);
	arrFloat = palloc0(sizeof(double*) * float_indices_cnt);

	for (i = 0; i < rows; i++)
	{
		HeapTuple   spi_tuple = SPI_tuptable->vals[i];
		int32 j;
		for (j = 0; j < len; j++)
			outnulls[j] = true;


		for (j = 0; j < SPI_natts; j++)
		{
			char    *value;
			value = SPI_getvalue(spi_tuple, SPI_tuptable->tupdesc, j+1);

			if (out_names[j] > -1)
			{
				outvalues[out_names[j]] = (Datum) cstring_to_text(strdup(value));
				outnulls[out_names[j]] = false;
			}

			if (iscategory[j] == ML_FEATURE_FLOAT)
			{
				double d;  
				sscanf(value, "%lf", &d);
				arrFloat[feature_idx[j]] = d;
				continue;
			}
			if (iscategory[j] == ML_FEATURE_CATEGORICAL)
			{
				arrCat[feature_idx[j]] = pstrdup(value);
				continue;
			}
		}

		if (!CalcModelPredictionSingle(modelHandle, arrFloat, float_indices_cnt,
								       arrCat, cat_indices_cnt, result_pa,
								       model_dimension))
		{
			elog( ERROR, "CalcModelPrediction error message: %s \nrow num=%d",
				GetErrorString(),i);

		}

		outnulls[len-1] = false;
		outvalues[len-1] = (Datum) cstring_to_text(sigmoid(result_pa[0]) > 0.5 ? "1":"0");		
		do_tup_output(tstate, outvalues, outnulls);
	}


	SPI_finish();




	
	end_tup_output(tstate);

	pfree(query.data);


	
	MemoryContextSwitchTo(oldCtx);
	

	free(cat_indices);
	free(float_indices);
	free(featureNames);

	ModelCalcerDelete(modelHandle);
	return;






	// /* end create tupledesc of out data*/


	// /* prepare for projection of tuples */
	// tstate = begin_tup_output_tupdesc(dest, tupdesc, &TTSOpsVirtual);

	// rel = table_open(PredictTableOid, AccessShareLock);
	// scan = table_beginscan(rel, GetLatestSnapshot(), 0, NULL);

	// while ((tup = heap_getnext(scan, ForwardScanDirection)) != NULL)
	// {
	// 	CHECK_FOR_INTERRUPTS();
	// 	if (!HeapTupleIsValid(tup))
	// 	{
	// 		elog(ERROR, " lookup failed for tuple");
	// 	}

	// 	/* Data row */
	// 	heap_deform_tuple(tup, 	tupdesc, values, nulls);

	// 	CreatePredictInputData(tupdesc, form->relnatts, arrTypes, values, &arrFloat, &arrCat);

	// 	if ( !CalcModelPredictionSingle(
	// 			modelHandle,
	// 			arrFloat, float_cnt,
	// 			(const char**) arrCat, cat_cnt,
	// 			result_pa, model_dimension)
	// 	   )
	// 	{
	// 		elog(ERROR, "prediction error in row %d: %s",j, GetErrorString());
	// 	}
	// 	max_probability = -1;
	// 	max_probability_idx = -1;
	// 	for(i=0; i < model_dimension; i++)
	// 	{
	// 		if (result_pa[i] > max_probability)
	// 		{
	// 			max_probability = result_pa[i];
	// 			max_probability_idx = i;
	// 		}
	// 	}

	// 	/* out row to output */
	// 	for (i=0; i < form->relnatts; i++)
	// 	{
	// 		outvalues[i] = values[i];
	// 		outnulls[i] = nulls[i];
	// 	}

	// 	/* out predict to output */
	// 	if ( model_type == 'C')
	// 	{
	// 		if (model_dimension == 1) // Logloss
	// 		{
	// 			if (classes_json_str)
	// 				outvalues[form->relnatts] = (Datum) cstring_to_text(classes[sigmoid(result_pa[0]) > 0.5 ? 1: 0]);
	// 			else
	// 			{
	// 				if (result_pa[0] > 0.5)
	// 					outvalues[form->relnatts] = (Datum) cstring_to_text( "1");
	// 				else
	// 					outvalues[form->relnatts] = (Datum) cstring_to_text( "0");
	// 			}
	// 		}
	// 		else//  Multiclass
	// 			outvalues[form->relnatts] = (Datum) cstring_to_text(classes[max_probability_idx]);
	// 	}
	// 	else
	// 	{
	// 		outvalues[form->relnatts] = (Datum) cstring_to_text(psprintf("%g", result_pa[0]));
	// 	}

	// 	do_tup_output(tstate, outvalues, outnulls);
	// }


	// do_tup_output(tstate, outvalues, outnulls);
	// end_tup_output(tstate);


	// pfree(arrTypes);	
	// table_close(rel, AccessShareLock);
	// table_endscan(scan);

	// if (classes)
	// {
	// 	pfree(classes[0]);
	// 	pfree(classes[1]);
	// 	pfree(classes);
	// }
	// ModelCalcerDelete(modelHandle);
	// // MemoryContextSwitchTo(oldcxt);
}


static void
SaveSidToMetadata( uint32 sid, char *modelname, ModelType modelType, char *str_parameter, char *queryString)
{
	TupleDesc   tupdesc;
	HeapTuple tup = NULL;
	Datum  *values;
	bool   *nulls, *doReplace;
	struct stat st;
	char model_type[2] = {'C', '\0'};
	Relation rel, idxrel;
	IndexScanDesc scan;
	TupleTableSlot* slot;
	ScanKeyData skey[1];
	bool found = false;
	NameData	name_name;

	namestrcpy(&name_name, modelname);

	modelTypeToChar(modelType, model_type);


	/* save metadata  */
	if (MetadataTableOid == InvalidOid)
	{
			MetadataTableOid  = get_relname_relid(ML_MODEL_METADATA,
												  PG_PUBLIC_NAMESPACE);
			MetadataTableIdxOid = get_relname_relid(ML_MODEL_METADATA_IDX,
													PG_PUBLIC_NAMESPACE);
	}

	tupdesc = GetMlModelTableDesc();

	values = (Datum*)palloc0( sizeof(Datum) * Natts_model);
	nulls = (bool *) palloc(sizeof(bool) * Natts_model);
	doReplace = (bool *) palloc0(sizeof(bool) * Natts_model);

	memset(nulls, true, Natts_model);

	/* Found by model name in ml_model */
 
	rel = table_open(MetadataTableOid, RowExclusiveLock);
	idxrel = index_open(MetadataTableIdxOid, AccessShareLock);

	scan = index_beginscan(rel, idxrel, GetTransactionSnapshot(), 1 /* nkeys */, 0 /* norderbys */);

	ScanKeyInit(&skey[0],
				Anum_ml_name ,
				BTGreaterEqualStrategyNumber, F_NAMEEQ,
				NameGetDatum(&name_name));

	index_rescan(scan, skey, 1, NULL /* orderbys */, 0 /* norderbys */);

	slot = table_slot_create(rel, NULL);
	found = false;
	while (index_getnext_slot(scan, ForwardScanDirection, slot))
	{
		bool should_free;
		tup = ExecFetchSlotHeapTuple(slot, false, &should_free);
		
		heap_deform_tuple(tup,  tupdesc, values, nulls);

		if(should_free) heap_freetuple(tup);
		found = true;
	}


	nulls[Anum_ml_model_type-1] = false;
	values[Anum_ml_model_type-1] = CStringGetTextDatum(model_type);
	doReplace[Anum_ml_model_type-1]  = true;

	if (str_parameter)
	{
		nulls[Anum_ml_model_args-1] = false;
		values[Anum_ml_model_args-1] = CStringGetTextDatum(str_parameter);
		doReplace[Anum_ml_model_args-1]  = true;
	}

	nulls[Anum_ml_model_sid-1] = false;
	values[Anum_ml_model_sid-1] = Int32GetDatum(sid);
	doReplace[Anum_ml_model_sid-1]  = true;

	nulls[Anum_ml_model_acc-1] = true;
	doReplace[Anum_ml_model_acc-1]  = true;

	nulls[Anum_ml_model_fieldlist-1] = true;
	doReplace[Anum_ml_model_fieldlist-1]  = true;

	nulls[Anum_ml_model_data-1] = true;
	doReplace[Anum_ml_model_data-1]  = true;


	elog(WARNING, "query: '%s'", queryString);
	if (queryString)
	{
		nulls[Anum_ml_model_query-1] = false;
		values[Anum_ml_model_query-1] = CStringGetTextDatum(queryString);
		doReplace[Anum_ml_model_query-1]  = true;
	}

	if (found)
	{
		tup = heap_modify_tuple(tup, tupdesc,values, nulls, doReplace);
		CatalogTupleUpdate(rel, &tup->t_self, tup);
	}

	index_endscan(scan);
	index_close(idxrel, AccessShareLock);
	ExecDropSingleTupleTableSlot(slot);

	if (!found)
	{
		nulls[Anum_ml_name-1] = false;
		values[Anum_ml_name-1] = CStringGetDatum(modelname);
		
		tup = heap_form_tuple(tupdesc, values, nulls);

		CatalogTupleInsert(rel, tup);
		heap_freetuple(tup);
	}
	
	table_close(rel, RowExclusiveLock);
}


static void SendRequest(int32 sid)
{

	CURL *curl;
	CURLcode res;
	long response_code;
	StringInfoData  buf;
	initStringInfo(&buf);

	curl_global_init(CURL_GLOBAL_DEFAULT);
	curl = curl_easy_init();

	if(curl)
	{
		curl_easy_setopt(curl, CURLOPT_URL, URL_REQUEST);
		curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");


		
		appendStringInfo(&buf, "{\"num\":%d}", sid);
		

		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buf.data);
		
		struct curl_slist *headers = NULL;
		headers = curl_slist_append(headers, "Content-Type: application/json");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);


		res = curl_easy_perform(curl);
		
		if(res != CURLE_OK)
		{
			elog(ERROR, "request to %s failed: %s", URL_REQUEST,
					curl_easy_strerror(res));
		}		
		else
		{
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
			elog(WARNING, "Response code: %ld\n", response_code);
		}

		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
	}
	
	curl_global_cleanup();
}



static int
LoadModelFromFileAndSaveToMetadata(const char* tmp_name, char* modelname,
				 ModelType modelType, Datum acc, char* str_parameter)
{
	TupleDesc   tupdesc;
	HeapTuple tup = NULL;
	Datum  *values;
	bool   *nulls, *doReplace;
	struct stat st;
	char model_type[2] = {'C', '\0'};
	int file_length;
	char *parms = NULL, *classes = NULL, *loss_function = NULL;
	void *model_buffer;
	int rc;
	Relation rel, idxrel;
	IndexScanDesc scan;
	TupleTableSlot* slot;
	ScanKeyData skey[1];
	bool found = false;
	NameData	name_name;


	namestrcpy(&name_name, modelname);

	modelTypeToChar(modelType, model_type);


	rc = stat(tmp_name, &st);
	if (rc)
	{
		const char * errmsg = strerror(errno);
		elog(ERROR, "Model file \"%s\" not found\n%s", tmp_name,
					errmsg);
	}

	file_length = st.st_size;

	// save metadata


	/* save metadata  */
	if (MetadataTableOid == InvalidOid)
	{
			MetadataTableOid  = get_relname_relid(ML_MODEL_METADATA,
												  PG_PUBLIC_NAMESPACE);
			MetadataTableIdxOid = get_relname_relid(ML_MODEL_METADATA_IDX,
													PG_PUBLIC_NAMESPACE);
	}

	tupdesc = GetMlModelTableDesc();

	values = (Datum*)palloc0( sizeof(Datum) * Natts_model);
	nulls = (bool *) palloc(sizeof(bool) * Natts_model);
	doReplace = (bool *) palloc0(sizeof(bool) * Natts_model);

	memset(nulls, true, Natts_model);

	/* Found by model name in ml_model */
 
	rel = table_open(MetadataTableOid, RowExclusiveLock);
	idxrel = index_open(MetadataTableIdxOid, AccessShareLock);

	scan = index_beginscan(rel, idxrel, GetTransactionSnapshot(), 1 /* nkeys */, 0 /* norderbys */);

	ScanKeyInit(&skey[0],
				Anum_ml_name ,
				BTGreaterEqualStrategyNumber, F_NAMEEQ,
				NameGetDatum(&name_name));

	index_rescan(scan, skey, 1, NULL /* orderbys */, 0 /* norderbys */);

	slot = table_slot_create(rel, NULL);
	found = false;
	while (index_getnext_slot(scan, ForwardScanDirection, slot))
	{
		bool should_free;
		tup = ExecFetchSlotHeapTuple(slot, false, &should_free);
		
		heap_deform_tuple(tup,  tupdesc, values, nulls);

		if(should_free) heap_freetuple(tup);
		found = true;
	}

	nulls[Anum_ml_model_type-1] = false;
	values[Anum_ml_model_type-1] = CStringGetTextDatum(model_type);
	doReplace[Anum_ml_model_type-1]  = true;

	if (acc)
	{
		nulls[Anum_ml_model_acc-1] = false;
		values[Anum_ml_model_acc-1] = Float4GetDatum(DatumGetFloat8(acc));
		doReplace[Anum_ml_model_acc-1]  = true;
	}

	if (str_parameter)
	{
		nulls[Anum_ml_model_args-1] = false;
		values[Anum_ml_model_args-1] = CStringGetTextDatum(str_parameter);
		doReplace[Anum_ml_model_args-1]  = true;
	}


	nulls[Anum_ml_model_data-1] = false;
	values[Anum_ml_model_data-1] = LoadFileToBuffer(tmp_name, file_length, &model_buffer);
	doReplace[Anum_ml_model_data-1]  = true;

	nulls[Anum_ml_model_fieldlist-1] = false;
	values[Anum_ml_model_fieldlist-1] = GetFeaturesFieldInfo(model_buffer, file_length, &parms, &classes);
	doReplace[Anum_ml_model_fieldlist-1]  = true;

	nulls[Anum_ml_model_info-1] = false;
	values[Anum_ml_model_info-1] = CStringGetTextDatum(parms);
	doReplace[Anum_ml_model_info-1]  = true;

	if (modelType == MODEL_TYPE_CLASSIFICATION && classes != NULL)
	{
		nulls[Anum_ml_model_classes-1] = false;
		values[Anum_ml_model_classes-1] = CStringGetTextDatum(classes);
		doReplace[Anum_ml_model_classes-1]  = true;
	}

	loss_function = GetLossFunctionFromParms(parms);
	nulls[Anum_ml_model_loss_function-1] = false;
	values[Anum_ml_model_loss_function-1] = CStringGetTextDatum(loss_function);
	doReplace[Anum_ml_model_loss_function-1]  = true;


	if (found)
	{
		tup = heap_modify_tuple(tup, tupdesc,values, nulls, doReplace);
		CatalogTupleUpdate(rel, &tup->t_self, tup);
	}
	

	index_endscan(scan);
	index_close(idxrel, AccessShareLock);
	ExecDropSingleTupleTableSlot(slot);

	if (!found)
	{
		nulls[Anum_ml_name-1] = false;
		values[Anum_ml_name-1] = CStringGetDatum(modelname);
		
		tup = heap_form_tuple(tupdesc, values, nulls);

		CatalogTupleInsert(rel, tup);
		heap_freetuple(tup);
	}
	
	table_close(rel, RowExclusiveLock);


	return rc;
}


/*
 * Load model from file
 */
void
LoadModelExecuteStmt(LoadModelStmt *stmt)
{
	if (RecoveryInProgress())
	{
		ereport(ERROR,
				(errcode(ERRCODE_WITH_CHECK_OPTION_VIOLATION),
				 errmsg("CREATE statement accepted only master, it is replication"),
				 errhint("You might need create the model in master")));


	}


	LoadModelFromFileAndSaveToMetadata(stmt->filename, stmt->modelname, MODEL_TYPE_UNDEFINED, 0, NULL);
}


/*
 * Model accuratly
 name          | name         | modelName имя модели 
 model_file    | text         |   скорее всего не испрользуется
 model_type    | character(1) | modelType
 acc           | real         |
 info          | text         | параметры
 args          | text         | аргументы 
 data          | bytea        | данные файла модели
 classes       | text         | классы модели
 loss_function | text         | loss функция
 sid           | integer      | sid - рандомное число или NULL  
 query         | text         | SELECT запрос на датасет

При выполнении оператора CREATE MODEL записывается
 - имя  stmt->modelname
 - тип  stmt->modelclass
 - запрос на датасет 
 - аргументы для машинного обучения (str_parameter)
  - sid некое rand число


 */
void
CreateModelExecute2Stmt(CreateModelStmt *stmt, const char *queryString, DestReceiver *dest)
{
	char *p = NULL;
	int32 sid;

	if (RecoveryInProgress())
	{
		ereport(ERROR,
				(errcode(ERRCODE_WITH_CHECK_OPTION_VIOLATION),
				 errmsg("CREATE statement accepted only master, it is replication"),
				 errhint("You might need create the model in master")));
	}


	SelectStmt  *select = stmt->query;
	List *targetList = select->targetList;
	Assert( targetList != NULL);

	ListCell *cell = list_head(targetList);
	ResTarget *target = lfirst_node(ResTarget, cell);

	p = pstrdup(queryString + target->location);
	elog(WARNING, "target: SELECT %s", p );
	srand(time(NULL));
	sid = rand();
	elog(WARNING, "sid: %d", sid );

	char *args = CreateJsonModelParameters(stmt);
	SaveSidToMetadata( sid,	stmt->modelname,
			stmt->modelclass, args, p);
	SendRequest(sid);
	pfree(p);
}

/*
 * Model accuratly
 */
void
CreateModelExecuteStmt(CreateModelStmt *stmt, DestReceiver *dest)
{
	Datum acc;
	TupleDesc   tupdesc;
	TupOutputState *tstate;
	NameData	name_name;
	const char* tmp_name = tempnam("/tmp/", "cbm_");
	char *str_parameter;
	int rc;
	char *res_out;

	if (RecoveryInProgress())
	{
		ereport(ERROR,
				(errcode(ERRCODE_WITH_CHECK_OPTION_VIOLATION),
				 errmsg("CREATE statement accepted only master, it is replication"),
				 errhint("You might need create the model in master")));


	}


	namestrcpy(&name_name, stmt->modelname);

	str_parameter = CreateJsonModelParameters(stmt);


	if (mlJsonWrapperOid == InvalidOid)
	{
		mlJsonWrapperOid = GetProcOidByName(ML_MODEL_LEARN_FUNCTION);
	}
	acc = OidFunctionCall5(  mlJsonWrapperOid,
					CStringGetTextDatum(stmt->modelname),
					Int32GetDatum(stmt->modelclass),
					CStringGetTextDatum(str_parameter),
					CStringGetTextDatum(stmt->tablename),
					CStringGetTextDatum(tmp_name));


	rc = LoadModelFromFileAndSaveToMetadata(tmp_name, stmt->modelname, stmt->modelclass, acc, str_parameter);


	/* need a tuple descriptor representing a single TEXT column */
	tupdesc = GetCreateModelResultDesc();

	/* prepare for projection of tuples */
	tstate = begin_tup_output_tupdesc(dest, tupdesc, &TTSOpsVirtual);

	/* Send it */
	res_out = psprintf("%g", DatumGetFloat8(acc));

	do_text_output_oneline(tstate, res_out);
	end_tup_output(tstate);

	if (rc == 0)
	{
		remove(tmp_name);
	}
}

Oid
GetProcOidByName(const char* proname)
{
	Oid found_oid = -1;
	Relation rel, idxrel;
	NameData fname;
	IndexScanDesc scan;
	TupleTableSlot* slot;
	HeapTuple tup;
	ScanKeyData skey[1];
	Name name = &fname;

	namestrcpy(name, proname);

	rel = table_open(ProcedureRelationId, RowExclusiveLock);
	idxrel = index_open(ProcedureOidIndexId, AccessShareLock);


	scan = index_beginscan(rel, idxrel, GetTransactionSnapshot(), 1 /* nkeys */, 0 /* norderbys */);

	ScanKeyInit(&skey[0],
				Anum_pg_proc_proname,
				BTGreaterEqualStrategyNumber, F_NAMEEQ,
				NameGetDatum(name));


	index_rescan(scan, skey, 1, NULL /* orderbys */, 0 /* norderbys */);

	slot = table_slot_create(rel, NULL);
	while (index_getnext_slot(scan, ForwardScanDirection, slot))
	{
		Form_pg_proc record;
		bool should_free;

		tup = ExecFetchSlotHeapTuple(slot, false, &should_free);
		record = (Form_pg_proc) GETSTRUCT(tup);

		if(strcmp(record->proname.data, name->data) == 0)
		{
			found_oid = record->oid;
			if(should_free) heap_freetuple(tup);
			break;
		}

		if(should_free) heap_freetuple(tup);
	}

	index_endscan(scan);
	ExecDropSingleTupleTableSlot(slot);

	index_close(idxrel, AccessShareLock);
	table_close(rel, RowExclusiveLock);

	if (found_oid == InvalidOid)
		elog(ERROR, "procedure %s not found", proname);

	return found_oid;
}


/**
 * GetLossFunctionFromParms() Return type of loss function
 *
 * input params - json string from model params
 **/
static char*
GetLossFunctionFromParms(char* parms)
{
	Jsonb *j;
	char *p = NULL;

	Datum dt_buffer  = CStringGetDatum(parms);
	Datum res = DirectFunctionCall1(jsonb_in, dt_buffer);
	j = DatumGetJsonbP(res);

	if (JB_ROOT_IS_OBJECT(j))
	{
		JsonbIterator *it;
		JsonbIteratorToken type;
		JsonbValue  jb;
		bool isFinish = false;
		enum ml_class_state_t classNamesState = ML_STATE_NONE;


		it = JsonbIteratorInit(&j->root);
		while ((type = JsonbIteratorNext(&it, &jb, false))
			   != WJB_DONE)
		{
			switch(jb.type)
			{
			case jbvString:
				if (classNamesState == ML_STATE_KEY && type == WJB_VALUE)
				{
					isFinish = true;
					p = pnstrdup(jb.val.string.val, jb.val.string.len);
					break;
				}
				if (type == WJB_KEY && strncmp(jb.val.string.val, "loss_function", 13) == 0)
				{
					classNamesState = ML_STATE_KEY;
				}
				else
				{
					classNamesState = ML_STATE_NONE;
				}
				break;

			default:
				;
			}

			if (isFinish)
				break;
		}
	}
	return p;
}


static char**
GetClassesFromJson(char* classes_json_str)
{
	char **p, *pp;
	Jsonb *j;

	Datum dt_buffer  = CStringGetDatum(classes_json_str);
	Datum res = DirectFunctionCall1(jsonb_in, dt_buffer);
	j = DatumGetJsonbP(res);

	if (JB_ROOT_IS_OBJECT(j))
	{
		JsonbIterator *it;
		JsonbIteratorToken type;
		JsonbValue  jb;
		int32 nElems, i;
		bool isFinish = false;
		enum ml_class_state_t classNamesState = ML_STATE_NONE;

		it = JsonbIteratorInit(&j->root);
		while ((type = JsonbIteratorNext(&it, &jb, false))
			   != WJB_DONE)
		{
			switch(jb.type)
			{
			case jbvString:

				if (classNamesState == ML_STATE_NONE && strncmp(jb.val.string.val, "class_names", 11) == 0)
				{
					classNamesState = ML_STATE_KEY;
				}
				else if (classNamesState == ML_STATE_BEG_ARRAY)
				{
					p[i++] = pnstrdup(jb.val.string.val, jb.val.string.len);
					if (i > nElems)
						isFinish = true;
				}
				else
					classNamesState = ML_STATE_NONE;
				break;

			case jbvArray:
				if (classNamesState == ML_STATE_KEY && type == WJB_BEGIN_ARRAY)
				{
					classNamesState = ML_STATE_BEG_ARRAY;
					nElems = jb.val.array.nElems;
					p = (char**) palloc(sizeof(char*) * nElems);
					i = 0;
				}
				if (classNamesState == ML_STATE_BEG_ARRAY && type == WJB_END_ARRAY)
				{
					classNamesState = ML_STATE_NONE;
				}
				break;

			case jbvNumeric:
				if (classNamesState == ML_STATE_BEG_ARRAY)
				{
					pp = numeric_to_cstring(jb.val.numeric); // allocate ??
					p[i++] = pstrdup(pp);
					if (i > nElems)
						isFinish = true;
				}
			default:
				;
			}

			if (isFinish)
				break;
		}
	}	

	return p;
}

static ModelCalcerHandle*
GetMlModelByName(const char * name, char** classes_json_str, char **loss_function, char* model_out_type)
{
	Relation rel, idxrel;
	ScanKeyData skey[1];
	IndexScanDesc scan;
	NameData name_data;
	TupleTableSlot* slot;
	Datum  *values;
	bool   *nulls;
	TupleDesc tupdesc;


	bool found = false;
	namestrcpy(&name_data, name);
	
	
	if (MetadataTableOid == InvalidOid)
	{
			MetadataTableOid  = get_relname_relid(ML_MODEL_METADATA, PG_PUBLIC_NAMESPACE);
			MetadataTableIdxOid = get_relname_relid(ML_MODEL_METADATA_IDX, PG_PUBLIC_NAMESPACE);
	}

	values = (Datum*)palloc0( sizeof(Datum) * Natts_model);
	nulls = (bool *) palloc0(sizeof(bool) * Natts_model);
	tupdesc = GetMlModelTableDesc();

	rel = table_open(MetadataTableOid, RowExclusiveLock);
	idxrel = index_open(MetadataTableIdxOid, AccessShareLock);

	scan = index_beginscan(rel, idxrel, GetTransactionSnapshot(), 1 /* nkeys */, 0 /* norderbys */);

	ScanKeyInit(&skey[0],
				Anum_ml_name ,
				BTGreaterEqualStrategyNumber, F_NAMEEQ,
				NameGetDatum(&name_data));

	index_rescan(scan, skey, 1, NULL /* orderbys */, 0 /* norderbys */);

	slot = table_slot_create(rel, NULL);
	while (index_getnext_slot(scan, ForwardScanDirection, slot))
	{
		HeapTuple tup;
		bool should_free;
	
		tup = ExecFetchSlotHeapTuple(slot, false, &should_free);
		
		heap_deform_tuple(tup,  tupdesc, values, nulls);
		if(should_free) heap_freetuple(tup);
		found = true;
	}

	index_endscan(scan);
	index_close(idxrel, AccessShareLock);
	table_close(rel, RowExclusiveLock);

	ExecDropSingleTupleTableSlot(slot);

	if (found)
	{
		ModelCalcerHandle *modelHandle = ModelCalcerCreate();
		bytea	   *bstr = DatumGetByteaPP(values[Anum_ml_model_data-1]);
		text *txt, *type_text = DatumGetTextPP(values[Anum_ml_model_type-1]);
		int len = VARSIZE(bstr);
		const void* bufferData = VARDATA(bstr);
		const char* model_type = text_to_cstring(type_text);

		// *loss_function = text_to_cstring(DatumGetTextPP(values[Anum_ml_model_loss_function-1]));
		// *model_out_type = model_type[0];
		// if (model_type[0] == 'C' && !nulls[Anum_ml_model_classes-1])
		// {
		// 	txt  = DatumGetTextPP(values[Anum_ml_model_classes-1]);
		// 	if (txt)
		// 		*classes_json_str = text_to_cstring(txt);
		// }
		if (!LoadFullModelFromBuffer(modelHandle, bufferData, len))
		{
			elog(ERROR, "LoadFullModelFromBuffer error message: %s", GetErrorString());
		}
		
		elog(WARNING, "model type %s, len=%d", model_type, len);
		return modelHandle;
	}

	elog(ERROR, "name:%s found=%d", name, found);
}

static char *
read_whole_file(const char *filename, int *length)
{
	char	   *buf;
	FILE	   *file;
	size_t		bytes_to_read;
	struct stat fst;

	if (stat(filename, &fst) < 0)
		ereport(ERROR,
		(errcode_for_file_access(),
		 errmsg("could not stat file \"%s\": %m", filename)));

	if (fst.st_size > (MaxAllocSize - 1))
		ereport(ERROR,
		(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
		 errmsg("file \"%s\" is too large", filename)));
			bytes_to_read = (size_t) fst.st_size;

	if ((file = AllocateFile(filename, PG_BINARY_R)) == NULL)
		ereport(ERROR,
		(errcode_for_file_access(),
		 errmsg("could not open file \"%s\" for reading: %m",
				filename)));

	buf = (char *) palloc(bytes_to_read + 1);

	*length = fread(buf, 1, bytes_to_read, file);

	if (ferror(file))
		ereport(ERROR,
		(errcode_for_file_access(),
		 errmsg("could not read file \"%s\": %m", filename)));

	FreeFile(file);

	buf[*length] = '\0';
	return buf;
}

/*
 * Drop model from metadata
 */
void
DropModelExecuteStmt(DropModelStmt *stmt)
{
	Relation rel, idxrel;
	ScanKeyData skey[1];
	IndexScanDesc scan;
	NameData name_data;
	TupleTableSlot* slot;
	Datum  *values;
	bool   *nulls;
	TupleDesc tupdesc;
	bool found = false;

	if (RecoveryInProgress())
	{
		ereport(ERROR,
				(errcode(ERRCODE_WITH_CHECK_OPTION_VIOLATION),
				 errmsg("DROP statement accepted only master, it is replication"),
				 errhint("You might need create the model in master")));
	}

	namestrcpy(&name_data, stmt->modelname);
	
	
	if (MetadataTableOid == InvalidOid)
	{
			MetadataTableOid  = get_relname_relid(ML_MODEL_METADATA, PG_PUBLIC_NAMESPACE);
			MetadataTableIdxOid = get_relname_relid(ML_MODEL_METADATA_IDX, PG_PUBLIC_NAMESPACE);
	}

	values = (Datum*)palloc0( sizeof(Datum) * Natts_model);
	nulls = (bool *) palloc0(sizeof(bool) * Natts_model);
	tupdesc = GetMlModelTableDesc();

	rel = table_open(MetadataTableOid, RowExclusiveLock);
	idxrel = index_open(MetadataTableIdxOid, AccessShareLock);

	scan = index_beginscan(rel, idxrel, GetTransactionSnapshot(), 1 /* nkeys */, 0 /* norderbys */);

	ScanKeyInit(&skey[0],
				Anum_ml_name ,
				BTGreaterEqualStrategyNumber, F_NAMEEQ,
				NameGetDatum(&name_data));

	index_rescan(scan, skey, 1, NULL /* orderbys */, 0 /* norderbys */);

	slot = table_slot_create(rel, NULL);
	while (index_getnext_slot(scan, ForwardScanDirection, slot))
	{
		HeapTuple tup;
		bool should_free;
	
		tup = ExecFetchSlotHeapTuple(slot, false, &should_free);

		heap_deform_tuple(tup,  tupdesc, values, nulls);
		if(should_free) heap_freetuple(tup);
		CatalogTupleDelete(rel, &tup->t_self);		
		found = true;
		break;
	}

	index_endscan(scan);
	ExecDropSingleTupleTableSlot(slot);
	index_close(idxrel, AccessShareLock);
	table_close(rel, RowExclusiveLock);

	if (!found)
		elog(ERROR, "model %s not found", stmt->modelname);
}