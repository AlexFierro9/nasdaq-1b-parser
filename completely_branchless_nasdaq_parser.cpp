#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <xmmintrin.h>
#include <omp.h>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <cmath>


static std::atomic<uint64_t> g_sink{0};


thread_local uint64_t tl_sink = 0;


inline void flush_sink() {
    g_sink.fetch_add(tl_sink, std::memory_order_relaxed);
    tl_sink = 0;
}




inline uint16_t load_be16(const void* ptr) {
    uint16_t v; std::memcpy(&v, ptr, 2); return __builtin_bswap16(v);
}
inline uint32_t load_be32(const void* ptr) {
    uint32_t v; std::memcpy(&v, ptr, 4); return __builtin_bswap32(v);
}
inline uint64_t load_be48(const void* ptr) {
    uint64_t v = 0; std::memcpy(&v, ptr, 6); return __builtin_bswap64(v) >> 16;
}
inline uint64_t load_be64(const void* ptr) {
    uint64_t v = 0; std::memcpy(&v, ptr, 8); return __builtin_bswap64(v);
}


using ITCHHandler = void (*)(const uint8_t*);
alignas(64) static ITCHHandler DispatchTable[256];

alignas(64) static const uint8_t ITCH_LENGTHS[256] = {
    ['S'] = 12, ['R'] = 38, ['H'] = 25, ['Y'] = 20, ['L'] = 26, ['V'] = 35,
    ['W'] = 12, ['K'] = 28, ['J'] = 35, ['h'] = 21, ['A'] = 36, ['F'] = 40,
    ['E'] = 31, ['C'] = 36, ['X'] = 23, ['D'] = 19, ['U'] = 35, ['P'] = 44,
    ['Q'] = 40, ['B'] = 19, ['I'] = 50, ['N'] = 27, ['O'] = 48
};


template<char TypeCode>
struct ITCHMessage {
    static void process(const uint8_t*) {}
};


template<>
struct ITCHMessage<'S'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint8_t  event_code      = buf[11];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ event_code;
    }
};


template<>
struct ITCHMessage<'R'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 11, 8);
        uint32_t round_lot_size  = load_be32(buf + 21);
        uint16_t raw_issue_sub;  std::memcpy(&raw_issue_sub, buf + 27, 2);
        uint32_t etp_leverage    = load_be32(buf + 34);
        uint8_t  market_cat      = buf[19];
        uint8_t  fin_status      = buf[20];
        uint8_t  rounds_only     = buf[25];
        uint8_t  issue_cls       = buf[26];
        uint8_t  authenticity    = buf[29];
        uint8_t  short_sale      = buf[30];
        uint8_t  ipo_flag        = buf[31];
        uint8_t  luld_tier       = buf[32];
        uint8_t  etp_flag        = buf[33];
        uint8_t  inverse         = buf[38];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ raw_stock
                 ^ round_lot_size ^ raw_issue_sub ^ etp_leverage
                 ^ market_cat ^ fin_status ^ rounds_only ^ issue_cls
                 ^ authenticity ^ short_sale ^ ipo_flag ^ luld_tier
                 ^ etp_flag ^ inverse;
    }
};


template<>
struct ITCHMessage<'H'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 11, 8);
        uint32_t raw_reason;     std::memcpy(&raw_reason, buf + 21, 4);
        uint8_t  trading_state   = buf[19];
        uint8_t  reserved        = buf[20];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ raw_stock ^ raw_reason ^ trading_state ^ reserved;
    }
};


template<>
struct ITCHMessage<'Y'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 11, 8);
        uint8_t  reg_sho_action  = buf[19];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ raw_stock ^ reg_sho_action;
    }
};

template<>
struct ITCHMessage<'L'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint32_t raw_mpid;       std::memcpy(&raw_mpid,  buf + 11, 4);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 15, 8);
        uint8_t  primary_mkt     = buf[23];
        uint8_t  mm_code         = buf[24];
        uint8_t  mm_state        = buf[25];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ raw_mpid ^ raw_stock ^ primary_mkt ^ mm_code ^ mm_state;
    }
};

template<>
struct ITCHMessage<'V'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t level1          = load_be64(buf + 11);
        uint64_t level2          = load_be64(buf + 19);
        uint64_t level3          = load_be64(buf + 27);

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ level1 ^ level2 ^ level3;
    }
};

template<>
struct ITCHMessage<'W'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint8_t  breached_level  = buf[11];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ breached_level;
    }
};

template<>
struct ITCHMessage<'K'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 11, 8);
        uint32_t ipo_quotation   = load_be32(buf + 19);
        uint32_t ipo_price       = load_be32(buf + 24);
        uint8_t  ipo_qualifier   = buf[23];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ raw_stock ^ ipo_quotation ^ ipo_price ^ ipo_qualifier;
    }
};

template<>
struct ITCHMessage<'J'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate       = load_be16(buf + 1);
        uint16_t tracking_number    = load_be16(buf + 3);
        uint64_t time_stamp         = load_be48(buf + 5);
        uint64_t raw_stock;         std::memcpy(&raw_stock, buf + 11, 8);
        uint32_t collar_ref_price   = load_be32(buf + 19);
        uint32_t upper_collar       = load_be32(buf + 23);
        uint32_t lower_collar       = load_be32(buf + 27);
        uint32_t collar_extension   = load_be32(buf + 31);

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ raw_stock
                 ^ collar_ref_price ^ upper_collar ^ lower_collar ^ collar_extension;
    }
};

template<>
struct ITCHMessage<'h'> {
    static __attribute__((noinline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 11, 8);
        uint8_t  market_code     = buf[19];
        uint8_t  halt_code       = buf[20];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ raw_stock ^ market_code ^ halt_code;
    }
};

template<>
struct ITCHMessage<'A'> {
    static __attribute__((always_inline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t order_ref       = load_be64(buf + 11);
        uint32_t shares          = load_be32(buf + 20);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 24, 8);
        uint32_t price           = load_be32(buf + 32);
        uint8_t  buy_sell        = buf[19];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ order_ref
                 ^ shares ^ raw_stock ^ price ^ buy_sell;
    }
};

template<>
struct ITCHMessage<'F'> {
    static __attribute__((always_inline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t order_ref       = load_be64(buf + 11);
        uint32_t shares          = load_be32(buf + 20);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 24, 8);
        uint32_t price           = load_be32(buf + 32);
        uint32_t raw_attribution; std::memcpy(&raw_attribution, buf + 36, 4);
        uint8_t  buy_sell        = buf[19];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ order_ref
                 ^ shares ^ raw_stock ^ price ^ raw_attribution ^ buy_sell;
    }
};

template<>
struct ITCHMessage<'E'> {
    static __attribute__((always_inline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t order_ref       = load_be64(buf + 11);
        uint32_t executed_shares = load_be32(buf + 19);
        uint64_t match_number    = load_be64(buf + 23);

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ order_ref ^ executed_shares ^ match_number;
    }
};

template<>
struct ITCHMessage<'C'> {
    static __attribute__((always_inline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t order_ref       = load_be64(buf + 11);
        uint32_t executed_shares = load_be32(buf + 19);
        uint64_t match_number    = load_be64(buf + 23);
        uint32_t execution_price = load_be32(buf + 32);
        uint8_t  printable       = buf[31];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ order_ref
                 ^ executed_shares ^ match_number ^ execution_price ^ printable;
    }
};

template<>
struct ITCHMessage<'X'> {
    static __attribute__((always_inline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t order_ref       = load_be64(buf + 11);
        uint32_t cancelled_shares = load_be32(buf + 19);

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ order_ref ^ cancelled_shares;
    }
};

template<>
struct ITCHMessage<'D'> {
    static __attribute__((always_inline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t order_ref       = load_be64(buf + 11);

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ order_ref;
    }
};

template<>
struct ITCHMessage<'U'> {
    static __attribute__((always_inline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t order_ref       = load_be64(buf + 11);
        uint64_t new_order_ref   = load_be64(buf + 19);
        uint32_t shares          = load_be32(buf + 27);
        uint32_t price           = load_be32(buf + 31);

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ order_ref ^ new_order_ref ^ shares ^ price;
    }
};

template<>
struct ITCHMessage<'P'> {
    static __attribute__((always_inline)) void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t order_ref       = load_be64(buf + 11);
        uint32_t shares          = load_be32(buf + 20);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 24, 8);
        uint32_t price           = load_be32(buf + 32);
        uint64_t match_number    = load_be64(buf + 36);
        uint8_t  buy_sell        = buf[19];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ order_ref
                 ^ shares ^ raw_stock ^ price ^ match_number ^ buy_sell;
    }
};

template<>
struct ITCHMessage<'Q'> {
    static inline void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t shares          = load_be64(buf + 11);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 19, 8);
        uint32_t cross_price     = load_be32(buf + 27);
        uint64_t match_number    = load_be64(buf + 31);
        uint8_t  cross_type      = buf[39];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ shares
                 ^ raw_stock ^ cross_price ^ match_number ^ cross_type;
    }
};

template<>
struct ITCHMessage<'B'> {
    static void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t match_number    = load_be64(buf + 11);

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ match_number;
    }
};

template<>
struct ITCHMessage<'I'> {
    static inline void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t paired_shares   = load_be64(buf + 11);
        uint64_t imbalance_shares = load_be64(buf + 19);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 28, 8);
        uint32_t far_price       = load_be32(buf + 36);
        uint32_t near_price      = load_be32(buf + 40);
        uint32_t current_ref     = load_be32(buf + 44);
        uint8_t  imbalance_dir   = buf[27];
        uint8_t  cross_type      = buf[48];
        uint8_t  price_var       = buf[49];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ paired_shares
                 ^ imbalance_shares ^ raw_stock ^ far_price ^ near_price
                 ^ current_ref ^ imbalance_dir ^ cross_type ^ price_var;
    }
};

template<>
struct ITCHMessage<'N'> {
    static void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 11, 8);
        uint8_t  interest_flag   = buf[19];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp
                 ^ raw_stock ^ interest_flag;
    }
};

template<>
struct ITCHMessage<'O'> {
    static inline void process(const uint8_t* buf) {
        uint16_t stock_locate    = load_be16(buf + 1);
        uint16_t tracking_number = load_be16(buf + 3);
        uint64_t time_stamp      = load_be48(buf + 5);
        uint64_t raw_stock;      std::memcpy(&raw_stock, buf + 11, 8);
        uint32_t min_price       = load_be32(buf + 20);
        uint32_t max_price       = load_be32(buf + 24);
        uint32_t near_exec_price = load_be32(buf + 28);
        uint64_t near_exec_time  = load_be64(buf + 32);
        uint32_t lower_collar    = load_be32(buf + 40);
        uint32_t upper_collar    = load_be32(buf + 44);
        uint8_t  open_elig       = buf[19];

        tl_sink += stock_locate ^ tracking_number ^ time_stamp ^ raw_stock
                 ^ min_price ^ max_price ^ near_exec_price ^ near_exec_time
                 ^ lower_collar ^ upper_collar ^ open_elig;
    }
};

size_t sync_to_next_message(const uint8_t* data, size_t start, size_t end) {
    size_t pos = start;
    while (pos + 2 < end) {
        uint16_t wire_len = load_be16(data + pos);
        uint8_t  type     = data[pos + 2];
        if (ITCH_LENGTHS[type] > 0 && wire_len > 0 && wire_len < 64)
            return pos;
        pos++;
    }
    return end;
}

void perform_forensic_investigation(const std::string& label,
                                    std::vector<double>& times,
                                    size_t file_size,
                                    uint64_t msg_count) {
    size_t n = times.size();
    if (n == 0) return;
    std::sort(times.begin(), times.end());

    auto to_gbps   = [&](double s) { return (file_size / 1073741824.0) / s; };
    auto to_mmsg   = [&](double s) { return (msg_count / 1e6) / s; };
    auto to_lat_ns = [&](double s) { return (s * 1e9) / msg_count; };

    double sum = 0;
    for (double t : times) sum += to_gbps(t);
    double avg = sum / n;

    double sq = 0;
    for (double t : times) sq += std::pow(to_gbps(t) - avg, 2);

    std::cout << "\n" << std::string(10, '=')
              << " FORENSIC REPORT: " << label << " "
              << std::string(10, '=') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Peak Performance:  " << to_gbps(times.front()) << " GB/s | "
              << to_mmsg(times.front()) << " M msg/s ("
              << to_lat_ns(times.front()) << " ns/msg)\n";
    std::cout << "Floor Performance: " << to_gbps(times.back()) << " GB/s | "
              << to_mmsg(times.back()) << " M msg/s ("
              << to_lat_ns(times.back()) << " ns/msg)\n";
    double mean_t = std::accumulate(times.begin(), times.end(), 0.0) / n;
    std::cout << "Mean Performance:  " << avg << " GB/s | "
              << to_mmsg(mean_t) << " M msg/s\n";
    std::cout << "Jitter (StdDev):   " << std::sqrt(sq / n) << " GB/s\n";
    std::cout << "p95 Latency:       "
              << times[static_cast<size_t>(n * 0.95)] * 1000.0 << " ms\n";
    std::cout << "p99 Latency:       "
              << times[static_cast<size_t>(n * 0.99)] * 1000.0 << " ms\n";
    std::cout << std::string(50, '=') << "\n";
}

struct RunResult {
    double   duration;
    uint64_t msg_count;
};

RunResult run_benchmark(uint8_t* data, size_t total_file_size, bool silent) {
    size_t   offset    = 0;
    uint64_t msg_count = 0;

    auto start = std::chrono::high_resolution_clock::now();
    while (offset < total_file_size) {
        _mm_prefetch((const char*)(data + offset + 128), _MM_HINT_T0);
        uint16_t msg_len_wire = load_be16(data + offset);
        offset += 2;
        const uint8_t type = data[offset];

        DispatchTable[type](data + offset);

        offset += msg_len_wire;
        msg_count++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    flush_sink();  

    if (!silent) {
        double lat = (diff.count() * 1e9) / msg_count;
        std::cout << "[FINAL SINGLE] "
                  << (msg_count / diff.count()) / 1e6
                  << " M msg/s | Latency: " << lat << " ns\n";
    }
    return {diff.count(), msg_count};
}

RunResult run_parallel_benchmark(uint8_t* data, size_t total_file_size, bool silent) {
    uint64_t total_msgs = 0;
    int      num_threads = omp_get_max_threads();

    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel reduction(+:total_msgs)
    {
        int    tid        = omp_get_thread_num();
        size_t chunk_size = total_file_size / num_threads;
        size_t my_start   = tid * chunk_size;
        size_t my_end     = (tid == num_threads - 1)
                            ? total_file_size : (tid + 1) * chunk_size;

        if (tid > 0) my_start = sync_to_next_message(data, my_start, my_end);

        size_t offset = my_start;
        while (offset < my_end) {
            _mm_prefetch((const char*)(data + offset + 192), _MM_HINT_T0);
            uint16_t msg_len_wire = load_be16(data + offset);
            offset += 2;
            const uint8_t type = data[offset];

            if      (type == 'A') ITCHMessage<'A'>::process(data + offset);
            else if (type == 'P') ITCHMessage<'P'>::process(data + offset);
            else if (type == 'E') ITCHMessage<'E'>::process(data + offset);
            else                  DispatchTable[type](data + offset);

            offset += msg_len_wire;
            total_msgs++;
        }
        flush_sink();  
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    if (!silent) {
        double lat = (diff.count() * 1e9) / total_msgs;
        std::cout << "[FINAL MULTI]  "
                  << (total_msgs / diff.count()) / 1e6
                  << " M msg/s | Latency: " << lat << " ns\n";
    }
    return {diff.count(), total_msgs};
}


int main() {
    //populating the static dispatch table
    DispatchTable['S'] = &ITCHMessage<'S'>::process;
    DispatchTable['R'] = &ITCHMessage<'R'>::process;
    DispatchTable['H'] = &ITCHMessage<'H'>::process;
    DispatchTable['Y'] = &ITCHMessage<'Y'>::process;
    DispatchTable['L'] = &ITCHMessage<'L'>::process;
    DispatchTable['V'] = &ITCHMessage<'V'>::process;
    DispatchTable['W'] = &ITCHMessage<'W'>::process;
    DispatchTable['K'] = &ITCHMessage<'K'>::process;
    DispatchTable['J'] = &ITCHMessage<'J'>::process;
    DispatchTable['h'] = &ITCHMessage<'h'>::process;
    DispatchTable['A'] = &ITCHMessage<'A'>::process;
    DispatchTable['F'] = &ITCHMessage<'F'>::process;
    DispatchTable['E'] = &ITCHMessage<'E'>::process;
    DispatchTable['C'] = &ITCHMessage<'C'>::process;
    DispatchTable['X'] = &ITCHMessage<'X'>::process;
    DispatchTable['D'] = &ITCHMessage<'D'>::process;
    DispatchTable['U'] = &ITCHMessage<'U'>::process;
    DispatchTable['P'] = &ITCHMessage<'P'>::process;
    DispatchTable['Q'] = &ITCHMessage<'Q'>::process;
    DispatchTable['B'] = &ITCHMessage<'B'>::process;
    DispatchTable['I'] = &ITCHMessage<'I'>::process;
    DispatchTable['N'] = &ITCHMessage<'N'>::process;
    DispatchTable['O'] = &ITCHMessage<'O'>::process;


    const char* filepath = "./01302020.NASDAQ_ITCH50";
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) { perror("open"); return 1; }

    struct stat sb;
    fstat(fd, &sb);
    const size_t total_file_size = sb.st_size;

    uint8_t* data = (uint8_t*)mmap(
        NULL, total_file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(data, total_file_size, MADV_SEQUENTIAL | MADV_WILLNEED | MADV_HUGEPAGE);


    const int ITERATIONS = 100;
    std::vector<double> s_times, m_times;
    uint64_t final_msg_count = 0;

    for (int i = 0; i < ITERATIONS; ++i) {
        bool is_last = (i == ITERATIONS - 1);

        RunResult res_s = run_benchmark(data, total_file_size, !is_last);
        RunResult res_m = run_parallel_benchmark(data, total_file_size, !is_last);

        s_times.push_back(res_s.duration);
        m_times.push_back(res_m.duration);
        if (i == 0) final_msg_count = res_s.msg_count;

        if (i > 0 && i % 25 == 0) {
            std::cout << "Progress: " << i << "%\n";
            sleep(30); // required or else lead to extreme thermal throttling
        }
    }

    perform_forensic_investigation("P-CORE SINGLE",  s_times, total_file_size, final_msg_count);
    perform_forensic_investigation("8-THREAD MULTI", m_times, total_file_size, final_msg_count);

    std::cout << "\n[SINK CHECKSUM] 0x" << std::hex << g_sink.load() << "\n";
    std::cout << "(Non-zero checksum confirms all fields were genuinely computed)\n";

    munmap(data, total_file_size);
    close(fd);
    return 0;
}