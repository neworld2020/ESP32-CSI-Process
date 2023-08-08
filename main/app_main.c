/* Wi-Fi CSI console Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

#include <errno.h>
#include <string.h>
#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_err.h"
#include "esp_console.h"

#include "esp_wifi.h"
#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "hal/uart_ll.h"

#include "esp_radar.h"
#include "csi_commands.h"

#include "esp_mac.h"
#include "esp_ota_ops.h"
#include "esp_netif.h"
#include "esp_chip_info.h"

#include "mbedtls/base64.h"

#define RECV_ESPNOW_CSI
#define CONFIG_LESS_INTERFERENCE_CHANNEL    11
#define RADAR_EVALUATE_SERVER_PORT          3232

static QueueHandle_t g_csi_info_queue    = NULL;
static const char *TAG                   = "app_main";

void print_device_info()
{
    esp_chip_info_t chip_info = {0};
    const char *chip_name = NULL;
    const esp_app_desc_t *app_desc = esp_ota_get_app_description();
    esp_netif_ip_info_t local_ip   = {0};
    wifi_ap_record_t ap_info       = {0};

    esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &local_ip);
    esp_chip_info(&chip_info);
    esp_wifi_sta_get_ap_info(&ap_info);

    switch (chip_info.model) {
        case CHIP_ESP32:
            chip_name = "ESP32";
            break;

        case CHIP_ESP32S2:
            chip_name = "ESP32-S2";
            break;

        case CHIP_ESP32S3:
            chip_name = "ESP32-S3";
            break;

        case CHIP_ESP32C3:
            chip_name = "ESP32-C3";
            break;

        default:
            chip_name = "Unknown";
            break;
    }

    printf("DEVICE_INFO,%u,%s %s,%s,%d,%s,%s,%d,%d,%s,"IPSTR",%u\n",
           esp_log_timestamp(), app_desc->date, app_desc->time, chip_name,
           chip_info.revision, app_desc->version, app_desc->idf_ver,
           heap_caps_get_total_size(MALLOC_CAP_DEFAULT), esp_get_free_heap_size(),
           ap_info.ssid, IP2STR(&local_ip.ip), RADAR_EVALUATE_SERVER_PORT);
}

static void wifi_init(void)
{
    esp_err_t ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }

    ESP_ERROR_CHECK(ret);

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_NULL));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));

#ifdef RECV_ESPNOW_CSI 
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
#endif
}
static struct {
    struct arg_lit *train_start;
    struct arg_lit *train_stop;
    struct arg_lit *train_add;
    struct arg_str *predict_someone_threshold;
    struct arg_str *predict_move_threshold;
    struct arg_int *predict_buff_size;
    struct arg_int *predict_outliers_number;
    struct arg_str *collect_taget;
    struct arg_int *collect_number;
    struct arg_int *collect_duration;
    struct arg_lit *csi_start;
    struct arg_lit *csi_stop;
    struct arg_str *csi_output_type;
    struct arg_str *csi_output_format;
    struct arg_end *end;
} radar_args;

static struct console_input_config {
    bool train_start;
    float predict_someone_threshold;
    float predict_move_threshold;
    uint32_t predict_buff_size;
    uint32_t predict_outliers_number;
    char collect_taget[16];
    uint32_t collect_number;
    char csi_output_type[16];
    char csi_output_format[16];
} g_console_input_config = {
    .predict_someone_threshold = 0.002,
    .predict_move_threshold    = 0.002,
    .predict_buff_size         = 5,
    .predict_outliers_number   = 2,
    .train_start               = false,
    .collect_taget             = "unknown",
    .csi_output_type           = "LLFT",
    .csi_output_format         = "decimal"
};

static TimerHandle_t g_collect_timer_handele = NULL;

static void collect_timercb(TimerHandle_t timer)
{
    g_console_input_config.collect_number--;

    if (!g_console_input_config.collect_number) {
        xTimerStop(g_collect_timer_handele, 0);
        xTimerDelete(g_collect_timer_handele, 0);
        g_collect_timer_handele = NULL;
        strcpy(g_console_input_config.collect_taget, "unknown");
        return;
    }
}

static int wifi_cmd_radar(int argc, char **argv)
{
    if (arg_parse(argc, argv, (void **) &radar_args) != ESP_OK) {
        arg_print_errors(stderr, radar_args.end, argv[0]);
        return ESP_FAIL;
    }

    if (radar_args.train_start->count) {
        if (!radar_args.train_add->count) {
            esp_radar_train_remove();
        }

        esp_radar_train_start();
        g_console_input_config.train_start = true;
    }

    if (radar_args.train_stop->count) {
        esp_radar_train_stop(&g_console_input_config.predict_someone_threshold,
                             &g_console_input_config.predict_move_threshold);
        g_console_input_config.predict_someone_threshold *= 1.1;
        g_console_input_config.predict_move_threshold    *= 1.1;
        g_console_input_config.train_start               = false;

        printf("RADAR_DADA,0,0,0,%.6f,0,0,%.6f,0\n", 
                g_console_input_config.predict_someone_threshold,
                g_console_input_config.predict_move_threshold);
    }

    if (radar_args.predict_move_threshold->count) {
        g_console_input_config.predict_move_threshold = atof(radar_args.predict_move_threshold->sval[0]);
    }

    if (radar_args.predict_someone_threshold->count) {
        g_console_input_config.predict_someone_threshold = atof(radar_args.predict_someone_threshold->sval[0]);
    }

    if (radar_args.predict_buff_size->count) {
        g_console_input_config.predict_buff_size = radar_args.predict_buff_size->ival[0];
    }

    if (radar_args.predict_outliers_number->count) {
        g_console_input_config.predict_outliers_number = radar_args.predict_outliers_number->ival[0];
    }

    if (radar_args.collect_taget->count && radar_args.collect_number->count && radar_args.collect_duration->count) {
        g_console_input_config.collect_number = radar_args.collect_number->ival[0];
        strcpy(g_console_input_config.collect_taget, radar_args.collect_taget->sval[0]);

        if (g_collect_timer_handele) {
            xTimerStop(g_collect_timer_handele, portMAX_DELAY);
            xTimerDelete(g_collect_timer_handele, portMAX_DELAY);
        }

        g_collect_timer_handele = xTimerCreate("collect", pdMS_TO_TICKS(radar_args.collect_duration->ival[0]),
                                               true, NULL, collect_timercb);
        xTimerStart(g_collect_timer_handele, portMAX_DELAY);
    }

    if (radar_args.csi_output_format->count) {
        strcpy(g_console_input_config.csi_output_format, radar_args.csi_output_format->sval[0]);
    }

    if (radar_args.csi_output_type->count) {
        wifi_radar_config_t radar_config = {0};
        esp_radar_get_config(&radar_config);

        if (!strcasecmp(radar_args.csi_output_type->sval[0], "NULL")) {
            radar_config.wifi_csi_filtered_cb = NULL;
        } else {
            void wifi_csi_raw_cb(const wifi_csi_filtered_info_t *info, void *ctx);
            radar_config.wifi_csi_filtered_cb = wifi_csi_raw_cb;
            strcpy(g_console_input_config.csi_output_type, radar_args.csi_output_type->sval[0]);
        }

        esp_radar_set_config(&radar_config);
    }

    if (radar_args.csi_start->count) {
        esp_radar_start();
    }

    if (radar_args.csi_stop->count) {
        esp_radar_stop();
    }

    return ESP_OK;
}

void cmd_register_radar(void)
{
    radar_args.train_start = arg_lit0(NULL, "train_start", "Start calibrating the 'Radar' algorithm");
    radar_args.train_stop  = arg_lit0(NULL, "train_stop", "Stop calibrating the 'Radar' algorithm");
    radar_args.train_add   = arg_lit0(NULL, "train_add", "Calibrate on the basis of saving the calibration results");

    radar_args.predict_someone_threshold = arg_str0(NULL, "predict_someone_threshold", "<0 ~ 1.0>", "Configure the threshold for someone");
    radar_args.predict_move_threshold    = arg_str0(NULL, "predict_move_threshold", "<0 ~ 1.0>", "Configure the threshold for move");
    radar_args.predict_buff_size         = arg_int0(NULL, "predict_buff_size", "1 ~ 100", "Buffer size for filtering outliers");
    radar_args.predict_outliers_number   = arg_int0(NULL, "predict_outliers_number", "<1 ~ 100>", "The number of items in the buffer queue greater than the threshold");

    radar_args.collect_taget    = arg_str0(NULL, "collect_tagets", "<0 ~ 20>", "Type of CSI data collected");
    radar_args.collect_number   = arg_int0(NULL, "collect_number", "sequence", "Number of times CSI data was collected");
    radar_args.collect_duration = arg_int0(NULL, "collect_duration", "duration", "Time taken to acquire one CSI data");

    radar_args.csi_start  = arg_lit0(NULL, "csi_start", "Start collecting CSI data from Wi-Fi");
    radar_args.csi_stop   = arg_lit0(NULL, "csi_stop", "Stop CSI data collection from Wi-Fi");
    radar_args.csi_output_type   = arg_str0(NULL, "csi_output_type", "<NULL, LLFT, HT-LFT, STBC-HT-LTF>", "Type of CSI data");
    radar_args.csi_output_format = arg_str0(NULL, "csi_output_format", "<decimal, base64>", "Format of CSI data");
    radar_args.end               = arg_end(8);

    const esp_console_cmd_t radar_cmd = {
        .command = "radar",
        .help = "Radar config",
        .hint = NULL,
        .func = &wifi_cmd_radar,
        .argtable = &radar_args
    };

    ESP_ERROR_CHECK(esp_console_cmd_register(&radar_cmd));
}

static void csi_data_process_task(void *arg)
{
    wifi_csi_filtered_info_t *info = NULL;
    char *buffer = malloc(8 * 1024);
    static uint32_t count = 0;

    while (xQueueReceive(g_csi_info_queue, &info, portMAX_DELAY)) {
        size_t len = 0;
        wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;

        if (!count) {
            ESP_LOGI(TAG, "================ CSI RECV ================");
            len += sprintf(buffer + len, "type,sequence,timestamp,taget_seq,taget,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data\n");
        }

        if (!strcasecmp(g_console_input_config.csi_output_type, "LLFT")) {
            info->valid_len = info->valid_llft_len;
        } else if (!strcasecmp(g_console_input_config.csi_output_type, "HT-LFT")) {
            info->valid_len = info->valid_llft_len + info->valid_ht_lft_len;
        } else if (!strcasecmp(g_console_input_config.csi_output_type, "STBC-HT-LTF")) {
            info->valid_len = info->valid_llft_len + info->valid_ht_lft_len + info->valid_stbc_ht_lft_len;
        }

        len += sprintf(buffer + len, "CSI_DATA,%d,%u,%u,%s," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%u,%d,%d,%d,%d,%d,",
                       count++, esp_log_timestamp(), g_console_input_config.collect_number, g_console_input_config.collect_taget,
                       MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
                       rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
                       rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
                       rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel, rx_ctrl->secondary_channel,
                       rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->rx_state, info->valid_len, 0);

        // ESP_LOGW(TAG, "csi_output_format: %s", g_console_input_config.csi_output_format);

        if (!strcasecmp(g_console_input_config.csi_output_format, "base64")) {
            size_t size = 0;
            mbedtls_base64_encode((uint8_t *)buffer + len, sizeof(buffer) - len, &size, (uint8_t *)info->valid_data, info->valid_len);
            len += size;
            len += sprintf(buffer + len, "\n");
        } else {
            len += sprintf(buffer + len, "\"[%d", info->valid_data[0]);

            for (int i = 1; i < info->valid_len; i++) {
                len += sprintf(buffer + len, ",%d", info->valid_data[i]);
            }

            len += sprintf(buffer + len, "]\"\n");
        }

        // printf("%s", buffer);
        free(info);
    }

    free(buffer);
    vTaskDelete(NULL);
}

void wifi_csi_raw_cb(const wifi_csi_filtered_info_t *info, void *ctx)
{
    wifi_csi_filtered_info_t *q_data = malloc(sizeof(wifi_csi_filtered_info_t) + info->valid_len);
    *q_data = *info;
    memcpy(q_data->valid_data, info->valid_data, info->valid_len);

    if (!g_csi_info_queue || xQueueSend(g_csi_info_queue, &q_data, 0) == pdFALSE) {
        ESP_LOGW(TAG, "g_csi_info_queue full");
        free(q_data);
    }
}

static void wifi_radar_cb(const wifi_radar_info_t *info, void *ctx)
{
    // No Need For Wifi Radar
    return;

    static float *s_buff_wander = NULL;
    static float *s_buff_jitter = NULL;

    if (!s_buff_wander) {
        s_buff_wander = calloc(100, sizeof(float));
    }

    if (!s_buff_jitter) {
        s_buff_jitter = calloc(100, sizeof(float));
    }

    static uint32_t s_buff_count          = 0;
    uint32_t buff_max_size      = g_console_input_config.predict_buff_size;
    uint32_t buff_outliers_num  = g_console_input_config.predict_outliers_number;
    uint32_t someone_count = 0;
    uint32_t move_count    = 0;
    bool room_status       = false;
    bool human_status      = false;

    s_buff_wander[s_buff_count % buff_max_size] = info->waveform_wander;
    s_buff_jitter[s_buff_count % buff_max_size] = info->waveform_jitter;
    s_buff_count++;

    if (s_buff_count < buff_max_size) {
        return;
    }

    for (int i = 0; i < buff_max_size; i++) {
        if (s_buff_wander[i] > g_console_input_config.predict_someone_threshold) {
            someone_count++;
        }

        if (s_buff_jitter[i] > g_console_input_config.predict_move_threshold) {
            move_count++;
        }
    }

    if (someone_count >= buff_outliers_num) {
        room_status = true;
    }

    if (move_count >= buff_outliers_num) {
        human_status = true;
    }

    static uint32_t s_count = 0;

    if (!s_count) {
        ESP_LOGI(TAG, "================ RADAR RECV ================");
        ESP_LOGI(TAG, "type,sequence,timestamp,waveform_wander,someone_threshold,someone_status,waveform_jitter,move_threshold,move_status\n");
    }

    char timestamp_str[32] = {0};
    sprintf(timestamp_str, "%u", esp_log_timestamp());

    if (ctx) {
        strncpy(timestamp_str, (char *)ctx, 31);
    }

    printf("RADAR_DADA,%d,%s,%.6f,%.6f,%d,%.6f,%.6f,%d\n",
           s_count++, timestamp_str,
           info->waveform_wander, g_console_input_config.predict_someone_threshold, room_status,
           info->waveform_jitter, g_console_input_config.predict_move_threshold, human_status);

    static uint32_t s_last_move_time = 0;
    static uint32_t s_last_someone_time = 0;


    if (room_status) {
        if (human_status) {
            ESP_LOGI(TAG, "Someone moved");
            s_last_move_time = esp_log_timestamp();
        } else if (esp_log_timestamp() - s_last_move_time > 3 * 1000) {
            ESP_LOGI(TAG, "Someone");
        }

        s_last_someone_time = esp_log_timestamp();
    }
}

/* Event handler for catching system events */
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        wifi_radar_config_t radar_config = {0};
        wifi_ap_record_t ap_info;

        esp_radar_get_config(&radar_config);

        radar_config.csi_config.lltf_en = true;
        radar_config.csi_config.htltf_en = false;
        radar_config.csi_config.stbc_htltf2_en = false;
        esp_wifi_sta_get_ap_info(&ap_info);
        memcpy(radar_config.filter_mac, ap_info.bssid, sizeof(ap_info.bssid));
        ESP_ERROR_CHECK(esp_wifi_get_mac(ESP_IF_WIFI_STA, radar_config.filter_dmac));

        esp_radar_set_config(&radar_config);

        print_device_info();

        esp_err_t ret   = ESP_OK;
        const char *ping = "ping -i 10";
        ESP_ERROR_CHECK(esp_console_run(ping, &ret));

        extern esp_err_t radar_evaluate_server(uint32_t port);
        radar_evaluate_server(RADAR_EVALUATE_SERVER_PORT);

#ifdef RECV_ESPNOW_CSI
        ESP_ERROR_CHECK(esp_wifi_set_promiscuous(false));
#endif
    }
}

void app_main(void)
{
    // set log level
    esp_log_level_set("*", ESP_LOG_INFO);
    ESP_LOGI(TAG, "Application Start");
    /**
     * @brief Register serial command
     */
    esp_console_repl_t *repl = NULL;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    esp_console_dev_uart_config_t uart_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    repl_config.prompt = "csi> ";
    

#if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32S2
    /**< Fix serial port garbled code due to high baud rate */
    uart_ll_set_sclk(UART_LL_GET_HW(CONFIG_ESP_CONSOLE_UART_NUM), UART_SCLK_APB);
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
    uart_ll_set_baudrate(UART_LL_GET_HW(CONFIG_ESP_CONSOLE_UART_NUM), CONFIG_ESP_CONSOLE_UART_BAUDRATE, CONFIG_ESP_CONSOLE_UART_BAUDRATE);
#else
    uart_ll_set_baudrate(UART_LL_GET_HW(CONFIG_ESP_CONSOLE_UART_NUM), CONFIG_ESP_CONSOLE_UART_BAUDRATE);
#endif
#endif

    cmd_register_system();
    ESP_LOGI(TAG, "Cmd Register System Finish");
    cmd_register_ping();
    ESP_LOGI(TAG, "Cmd Register Ping Finish");
    cmd_register_wifi_config();
    ESP_LOGI(TAG, "Cmd Register WiFi Finish");
    cmd_register_wifi_scan();
    ESP_LOGI(TAG, "Cmd Register WiFi Scan Finish");
    cmd_register_radar();
    ESP_LOGI(TAG, "Cmd Register Finish");
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uart_config, &repl_config, &repl));
    ESP_ERROR_CHECK(esp_console_start_repl(repl));

    /**
     * @brief Initialize Wi-Fi Radar
     */

    wifi_init();
    esp_radar_init();
    ESP_LOGI(TAG, "Wifi and CSI Initialize Finish");

    wifi_radar_config_t radar_config = WIFI_RADAR_CONFIG_DEFAULT();
    radar_config.wifi_radar_cb = wifi_radar_cb;
    memcpy(radar_config.filter_mac, "\x1a\x00\x00\x00\x00\x00", 6);
    esp_radar_set_config(&radar_config);

    esp_radar_start();

    /**
     * @brief Initialize CSI serial port printing task, Use tasks to avoid blocking wifi_csi_raw_cb
     */
    g_csi_info_queue = xQueueCreate(64, sizeof(void *));
    xTaskCreate(csi_data_process_task, "csi_data_print", 4 * 1024, NULL, 0, NULL);

    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));
}
