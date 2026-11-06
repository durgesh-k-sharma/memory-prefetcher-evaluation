#ifndef IP_STRIDE_FDP_H
#define IP_STRIDE_FDP_H

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "address.h"
#include "champsim.h"
#include "modules.h"
#include "msl/lru_table.h"

struct ip_stride_fdp : public champsim::modules::prefetcher {
  struct tracker_entry {
    champsim::address ip{};
    champsim::block_number last_cl_addr{};
    champsim::block_number::difference_type last_stride{};

    auto index() const
    {
      using namespace champsim::data::data_literals;
      return ip.slice_upper<2_b>();
    }
    auto tag() const
    {
      using namespace champsim::data::data_literals;
      return ip.slice_upper<2_b>();
    }
  };

  struct lookahead_entry {
    champsim::address address{};
    champsim::address::difference_type stride{};
    int degree = 0;
    int control_degree = 0;
  };

  constexpr static std::size_t TRACKER_SETS = 256;
  constexpr static std::size_t TRACKER_WAYS = 4;
  constexpr static int MAX_PREFETCH_DEGREE = 3;
  constexpr static uint64_t FEEDBACK_BATCH_ADDRESS_EPISODES = 4096;
  constexpr static uint64_t LOW_USEFULNESS_PERCENT = 30;
  constexpr static uint64_t HIGH_USEFULNESS_PERCENT = 45;

  std::optional<lookahead_entry> active_lookahead;
  champsim::msl::lru_table<tracker_entry> table{TRACKER_SETS, TRACKER_WAYS};
  std::unordered_map<uint64_t, int> pending_prefetch_degrees;

  int current_degree = MAX_PREFETCH_DEGREE;
  std::array<uint64_t, MAX_PREFETCH_DEGREE + 1> window_resolved{};
  std::array<uint64_t, MAX_PREFETCH_DEGREE + 1> window_useful{};
  uint64_t total_useful = 0;
  uint64_t total_issued = 0;
  uint64_t total_tracked_address_episodes = 0;
  uint64_t total_resolved_address_episodes = 0;
  uint64_t total_useful_address_episodes = 0;
  uint64_t feedback_windows = 0;
  uint64_t degree_increases = 0;
  uint64_t degree_decreases = 0;

  void update_feedback(int issued_degree);
  void resolve_prefetch(uint64_t cache_line, bool useful);

public:
  using champsim::modules::prefetcher::prefetcher;

  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  void prefetcher_cycle_operate();
  void prefetcher_final_stats();
};

#endif
