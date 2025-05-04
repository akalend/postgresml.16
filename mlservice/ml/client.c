#include <stdio.h>
#include <curl/curl.h>

int main(void) {
    CURL *curl;
    CURLcode res;
    long response_code;

    // Инициализация libcurl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if(curl) {
        // Устанавливаем URL
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8000/ml");
        
        // Устанавливаем метод PUT
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        
        // Устанавливаем данные для отправки (можно изменить на нужные)
        const char *data = "{\"num\":123}"; // Пример JSON данных
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);
        
        // Устанавливаем заголовки (можно добавить другие при необходимости)
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        
        // Выполняем запрос
        res = curl_easy_perform(curl);
        
        // Проверяем на ошибки
        if(res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n",
                    curl_easy_strerror(res));
        
        else
        {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            printf("\nResponse code: %ld\n", response_code);
        }
        // Освобождаем ресурсы
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
    
    curl_global_cleanup();
    return 0;
}