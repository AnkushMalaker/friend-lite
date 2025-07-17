/*
 * TCP Stream Header for Voice PE XMOS Audio
 * Enhanced networking support for high-quality audio streaming
 */

#include <lwip/sockets.h>
#include <lwip/netdb.h>
#include <lwip/tcp.h>
#include <errno.h>

// TCP socket options for real-time audio
#define AUDIO_STREAM_BUFFER_SIZE 8192
#define AUDIO_STREAM_TIMEOUT_MS 5000

// Audio format constants for XMOS Voice PE
#define XMOS_SAMPLE_RATE_48K 48000
#define XMOS_SAMPLE_RATE_16K 16000  
#define XMOS_BITS_PER_SAMPLE 32
#define XMOS_CHANNELS_STEREO 2
#define XMOS_CHANNELS_MONO 1

// Network performance tuning
#define TCP_AUDIO_NODELAY 1
#define TCP_AUDIO_KEEPALIVE 1
#define TCP_AUDIO_KEEPIDLE 5     // 5 seconds
#define TCP_AUDIO_KEEPINTVL 1    // 1 second intervals  
#define TCP_AUDIO_KEEPCNT 3      // 3 probes

// Audio quality metrics
struct xmos_audio_stats {
    uint32_t total_bytes_sent;
    uint32_t total_packets_sent;
    uint32_t connection_errors;
    uint32_t last_error_time;
    float avg_packet_size;
};

static struct xmos_audio_stats audio_stats = {0};

// Enhanced TCP socket setup for audio streaming
static int setup_audio_socket(int sockfd) {
    int opt = 1;
    
    // Disable Nagle's algorithm for low latency
    if (lwip_setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) {
        return -1;
    }
    
    // Enable keepalive for connection monitoring
    if (lwip_setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt)) < 0) {
        return -1;
    }
    
    // Set keepalive parameters
    int keepidle = TCP_AUDIO_KEEPIDLE;
    int keepintvl = TCP_AUDIO_KEEPINTVL; 
    int keepcnt = TCP_AUDIO_KEEPCNT;
    
    lwip_setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
    lwip_setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
    lwip_setsockopt(sockfd, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
    
    // Set receive/send buffer sizes for audio streaming
    int buffer_size = AUDIO_STREAM_BUFFER_SIZE;
    lwip_setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
    lwip_setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
    
    return 0;
}

// Update audio streaming statistics
static void update_audio_stats(size_t bytes_sent) {
    audio_stats.total_bytes_sent += bytes_sent;
    audio_stats.total_packets_sent++;
    audio_stats.avg_packet_size = (float)audio_stats.total_bytes_sent / audio_stats.total_packets_sent;
}

// Log audio statistics periodically
static void log_audio_stats(const char* tag) {
    if (audio_stats.total_packets_sent % 1000 == 0 && audio_stats.total_packets_sent > 0) {
        ESP_LOGI(tag, "Audio Stats: %lu bytes, %lu packets, %.1f avg size, %lu errors",
                 audio_stats.total_bytes_sent,
                 audio_stats.total_packets_sent, 
                 audio_stats.avg_packet_size,
                 audio_stats.connection_errors);
    }
}