#include <cstdint>

enum ProfilerTag {
  Setup = 0,
  IssueTMA,
  IssueMMA,
  WaitTMA,
  WaitMMA,
  WaitMainloop,
  WaitEpilogue,
  Epilogue,
};

__device__ inline
int64_t globaltimer() {
  int64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  return t;
}

struct ProfilerMeta {
  int warp_id;
  int cta_rank;
  int stage;
  int phase;
  int bid;
  int bid_m;
  int bid_n;
  int iter_k;
};

__device__ inline
ProfilerMeta make_profiler_meta(
  int warp_id = -1,
  int cta_rank = -1,
  int stage = -1,
  int phase = -1,
  int bid = -1,
  int bid_m = -1,
  int bid_n = -1,
  int iter_k = -1
) {
  return {warp_id, cta_rank, stage, phase, bid, bid_m, bid_n, iter_k};
}

struct Profiler {
  int64_t *data_ptr_;
  int sm_id_;
  int cnt_;

  __device__
  static uint64_t pack_u8(int value) {
    return static_cast<uint64_t>(value) & 0xFFull;
  }

  __device__
  static uint64_t pack_u16(int value) {
    return static_cast<uint64_t>(value) & 0xFFFFull;
  }

  __device__
  void init(int num_entries, int64_t *data_ptr, int bid) {
    data_ptr_ = data_ptr + bid * (1 + num_entries * 4);
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(sm_id_));
    cnt_ = 0;
  }

  __device__
  void start(ProfilerTag tag) {
    start(tag, make_profiler_meta());
  }

  __device__
  void start(ProfilerTag tag, ProfilerMeta meta) {
    data_ptr_[1 + cnt_ * 4 + 0] =
      pack_u16(sm_id_)
      | (pack_u8(meta.warp_id) << 16)
      | (pack_u8(meta.cta_rank) << 24)
      | (pack_u8(tag) << 32)
      | (pack_u8(meta.stage) << 40)
      | (pack_u8(meta.phase) << 48)
      ;
    data_ptr_[1 + cnt_ * 4 + 1] =
      pack_u16(meta.bid)
      | (pack_u16(meta.bid_m) << 16)
      | (pack_u16(meta.bid_n) << 32)
      | (pack_u16(meta.iter_k) << 48)
      ;
    data_ptr_[1 + cnt_ * 4 + 2] = globaltimer();
  }

  __device__
  void stop() {
    data_ptr_[1 + cnt_ * 4 + 3] = globaltimer() - data_ptr_[1 + cnt_ * 4 + 2];
    cnt_ += 1;
  }

  __device__
  void flush() {
    data_ptr_[0] = cnt_;
  }
};
