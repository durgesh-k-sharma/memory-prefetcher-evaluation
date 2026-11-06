#include "ip_stride_fdp.h"

#include <iostream>

#include "cache.h"

void ip_stride_fdp::update_feedback(int issued_degree)
{
  ++feedback_windows;
  const uint64_t useful_percent = window_useful[issued_degree] * 100 / window_resolved[issued_degree];
  if (useful_percent < LOW_USEFULNESS_PERCENT && current_degree == issued_degree && issued_degree > 1) {
    current_degree = issued_degree - 1;
    ++degree_decreases;
  } else if (useful_percent >= HIGH_USEFULNESS_PERCENT && current_degree == issued_degree && current_degree < MAX_PREFETCH_DEGREE) {
    ++current_degree;
    ++degree_increases;
  }

  window_resolved[issued_degree] = 0;
  window_useful[issued_degree] = 0;
}

void ip_stride_fdp::resolve_prefetch(uint64_t cache_line, bool useful)
{
  auto found = pending_prefetch_degrees.find(cache_line);
  if (found == pending_prefetch_degrees.end())
    return;

  const int issued_degree = found->second;
  pending_prefetch_degrees.erase(found);
  ++window_resolved[issued_degree];
  ++total_resolved_address_episodes;
  if (useful) {
    ++window_useful[issued_degree];
    ++total_useful_address_episodes;
  }

  if (window_resolved[issued_degree] >= FEEDBACK_BATCH_ADDRESS_EPISODES)
    update_feedback(issued_degree);
}

uint32_t ip_stride_fdp::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                                 uint32_t metadata_in)
{
  if (useful_prefetch) {
    ++total_useful;
    resolve_prefetch(champsim::block_number{addr}.to<uint64_t>(), true);
  }

  champsim::block_number cl_addr{addr};
  champsim::block_number::difference_type stride = 0;
  auto found = table.check_hit({ip, cl_addr, stride});

  if (found.has_value()) {
    stride = champsim::offset(found->last_cl_addr, cl_addr);
    if (stride != 0 && stride == found->last_stride)
      active_lookahead = {champsim::address{cl_addr}, stride, current_degree, current_degree};
  }

  table.fill({ip, cl_addr, stride});
  return metadata_in;
}

void ip_stride_fdp::prefetcher_cycle_operate()
{
  if (active_lookahead.has_value()) {
    auto [old_pf_address, stride, degree, control_degree] = active_lookahead.value();
    assert(degree > 0);

    champsim::address pf_address{champsim::block_number{old_pf_address} + stride};
    if (intern_->virtual_prefetch || champsim::page_number{pf_address} == champsim::page_number{old_pf_address}) {
      const bool mshr_under_light_load = intern_->get_mshr_occupancy_ratio() < 0.5;
      const bool success = prefetch_line(pf_address, mshr_under_light_load, 0);
      if (success) {
        ++total_issued;
        const uint64_t cache_line = champsim::block_number{pf_address}.to<uint64_t>();
        const auto [pending, inserted] = pending_prefetch_degrees.insert_or_assign(cache_line, control_degree);
        (void)pending;
        if (inserted)
          ++total_tracked_address_episodes;
        active_lookahead = {pf_address, stride, degree - 1, control_degree};
      }

      if (active_lookahead->degree == 0)
        active_lookahead.reset();
    } else {
      active_lookahead.reset();
    }
  }
}

uint32_t ip_stride_fdp::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr,
                                              uint32_t metadata_in)
{
  if (evicted_addr != champsim::address{})
    resolve_prefetch(champsim::block_number{evicted_addr}.to<uint64_t>(), false);

  if (!prefetch)
    resolve_prefetch(champsim::block_number{addr}.to<uint64_t>(), false);

  return metadata_in;
}

void ip_stride_fdp::prefetcher_final_stats()
{
  std::cout << "FDP_IP_STRIDE_STATS"
            << " final_degree=" << current_degree
            << " degree_increases=" << degree_increases
            << " degree_decreases=" << degree_decreases
            << " feedback_windows=" << feedback_windows
            << " total_useful_events=" << total_useful
            << " useful_address_episodes=" << total_useful_address_episodes
            << " accepted_prefetch_requests=" << total_issued
            << " tracked_address_episodes=" << total_tracked_address_episodes
            << " resolved_address_episodes=" << total_resolved_address_episodes
            << " feedback_batch_address_episodes=" << FEEDBACK_BATCH_ADDRESS_EPISODES
            << " low_usefulness_percent=" << LOW_USEFULNESS_PERCENT
            << " high_usefulness_percent=" << HIGH_USEFULNESS_PERCENT << '\n';
}
