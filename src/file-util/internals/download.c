#include<curl/curl.h>

int dl() {
    CURLversion arg;
    curl_version_info(arg);
}