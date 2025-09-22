/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/fp8_to_fp16.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class Schedule,
  class TileShape_,
  class ElementAOptionalTuple,
  class StrideA_,
  class ElementBOptionalTuple,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopIntelXeXMX16GroupMixedPrecision<Stages, Schedule>,
    TileShape_,
    ElementAOptionalTuple,
    StrideA_,
    ElementBOptionalTuple,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
private:
  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };

public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelXeXMX16GroupMixedPrecision<Stages, Schedule>;
  using WorkgroupTileShape = TileShape_;

  
  static_assert(cute::is_tuple<ElementAOptionalTuple>::value ^ cute::is_tuple<ElementBOptionalTuple>::value, 
    "Either A OR B must be a tuple. It must take the from {ElementOperand, [ElementScale],"
    "[ElementZero]}. Inputs in [] are optional.");

  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;

  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;

  using ElementMMA = cute::conditional_t<IsATransformed, ElementB, ElementA>;
  using ElementQuant = cute::conditional_t<IsATransformed, ElementA, ElementB>;

  using ElementScale = cute::conditional_t<IsATransformed, detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>, detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>>;
  using StrideScale = cute::conditional_t<IsATransformed, detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>, detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>>;

  using ElementZero = cute::conditional_t<IsATransformed, detail::deduce_mixed_width_dtype_t<3, ElementAOptionalTuple>, detail::deduce_mixed_width_dtype_t<3, ElementBOptionalTuple>>;
  using StrideZero = cute::conditional_t<IsATransformed, detail::deduce_mixed_width_dtype_t<4, ElementAOptionalTuple>, detail::deduce_mixed_width_dtype_t<4, ElementBOptionalTuple>>;

  // For cases where we can't have a void type, we can use this to allow the code to compile when the scale / zero is void.
  using NonVoidElementScale = cute::conditional_t<cute::is_void_v<ElementScale>, ElementMMA, ElementScale>;
  using NonVoidElementZero = cute::conditional_t<cute::is_void_v<ElementZero>, ElementMMA, ElementZero>;

  using NonVoidStrideScale = cute::conditional_t<cute::is_same_v<StrideScale, void>, cute::Stride<_1, int64_t, int64_t> *, StrideScale>;
  using NonVoidStrideZero = cute::conditional_t<cute::is_same_v<StrideZero, void>, cute::Stride<_1, int64_t, int64_t> *, StrideZero>;
  using InternalNonVoidStrideScale = cute::remove_pointer_t<NonVoidStrideScale>;
  using InternalNonVoidStrideZero = cute::remove_pointer_t<NonVoidStrideZero>;
  static constexpr auto zero_elements_packed_along_k = get<0>(InternalNonVoidStrideZero{});

  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using MmaType = typename TiledMma::ValTypeA; // ValTypeA and ValTypeB are always same and reflects MMA type on intel Xe
  using LargerElementType = std::conditional_t<(cute::sizeof_bits_v<ElementA> > cute::sizeof_bits_v<ElementB>),
                                               ElementA,
                                               ElementB>;

  static_assert(!cute::is_same_v<ElementA, ElementB>, "Mixed precision GEMM requires different types for A and B!");
  static_assert(std::is_same_v<TransformA, cute::identity>, "Transformation for A is not currently supported on Intel PVC");
  static_assert(std::is_same_v<TransformB, cute::identity>, "Transformation for B is not currently supported on Intel PVC");
  
private:
   
  static constexpr ConversionMode 
  get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } 
    else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    }
    else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales = KernelConversionMode == ConversionMode::ConvertAndScale ||
                                        KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;
  static constexpr bool ModeHasScalesZero = KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;
public:
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr auto BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr auto BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr auto BLK_K = get<2>(WorkgroupTileShape{});
  
  static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);
  using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;
  
  using GmemTiledCopyScale = typename scale_zero_copy_traits<NonVoidElementScale, SG_N>::type;
  using GmemTiledCopyZero = typename scale_zero_copy_traits<NonVoidElementZero, SG_N, InternalNonVoidStrideZero>::type;

  static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
  using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));

  using traits_load_A = Copy_Traits<GmemTiledCopyA, InternalStrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
  using val_layout_load_A = decltype(make_layout(shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
  using Copy_A = decltype(make_tiled_copy(atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));

  using traits_load_B = Copy_Traits<GmemTiledCopyB, InternalStrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
  using val_layout_load_B = decltype(make_layout(shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
  using Copy_B = decltype(make_tiled_copy(atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));

  using traits_load_scale = Copy_Traits<GmemTiledCopyScale, InternalNonVoidStrideScale>;
  using atom_load_scale = Copy_Atom<traits_load_scale, NonVoidElementScale>;
  using val_layout_load_scale = decltype(make_layout(shape_div(typename traits_load_scale::BlockShape{}, CopyThreadShapeRev{}))); 
  using Copy_Scale = decltype(make_tiled_copy(atom_load_scale{}, Layout<CopyThreadShapeRev>{}, val_layout_load_scale{}));
  
  using traits_load_zero = Copy_Traits<GmemTiledCopyZero, InternalNonVoidStrideScale>;
  using atom_load_zero = Copy_Atom<traits_load_zero, NonVoidElementZero>;
  using val_layout_load_zero = decltype(make_layout(shape_div(typename traits_load_zero::BlockShape{}, CopyThreadShapeRev{}))); 
  using Copy_Zero = decltype(make_tiled_copy(atom_load_zero{}, Layout<CopyThreadShapeRev>{}, val_layout_load_zero{}));

  using TensorMKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), make_shape(0,0,0), InternalStrideA{}));   //(m, k)
  using TensorS = decltype(make_tensor(make_gmem_ptr(static_cast<NonVoidElementScale const*>(nullptr)), make_shape(0,0,0), InternalNonVoidStrideScale{}));

  // Purpose of this struct is to create a pointer of required type
  // for creating the TensorNKL type
  template <typename T, typename Enable = void>
  struct GenPtrType;

  template <typename T>
  struct GenPtrType <T, cute::enable_if_t<sizeof_bits_v<T> < 8>> {
    static constexpr auto get_pointer() {
      // For int4_t type, subbyte_iterator does not accept nullptr,
      // so need to create a pointer of type int4_t to get this code
      // working.
      T* ptr;
      return cute::subbyte_iterator<const T>(ptr);
    }
  };

  template <typename T>
  struct GenPtrType <T, cute::enable_if_t<sizeof_bits_v<T> >= 8>> {
    static constexpr auto get_pointer() {
      return make_gmem_ptr(static_cast<T const *>(nullptr));
    }
  };

  using TensorNKL = decltype(make_tensor(GenPtrType<ElementB>::get_pointer(), make_shape(0,0,0), InternalStrideB{}));   //(n, k)
  using TensorZ = decltype(make_tensor(GenPtrType<NonVoidElementZero>::get_pointer(), make_shape(0,0,0), InternalNonVoidStrideScale{}));
  using MainloopTensors = cute::tuple<TensorMKL, TensorNKL, TensorS, TensorZ>;

  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    NonVoidElementScale const** ptr_S = nullptr;
    NonVoidStrideScale dS{};
    NonVoidElementZero const** ptr_Z = nullptr;
    NonVoidStrideZero dZ{};
    int group_size = 1;
  };

  struct Params {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    NonVoidElementScale const** ptr_S = nullptr;
    NonVoidStrideScale dS{};
    NonVoidElementZero const** ptr_Z = nullptr;
    NonVoidStrideZero dZ{};
    int group_size;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape,
                          Arguments const &args, void *workspace) {
    (void)workspace;
    return Params{args.ptr_A, args.dA, args.ptr_B, args.dB, args.ptr_S, args.dS, args.ptr_Z, args.dZ, args.group_size};
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;

    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    constexpr int min_batch_aligned_elements_A = batch_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_batch_aligned_elements_B = batch_alignment_bits / sizeof_bits<ElementB>::value;
    for (int i = 0; i < problem_shapes.groups(); i++) {
      auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
      auto [M,N,K,L] = problem_shape_MNKL;

      implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
      implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});

      if (L > 1) {
        implementable &= get<2>(InternalStrideA{}) % min_batch_aligned_elements_A == 0;
        implementable &= get<2>(InternalStrideB{}) % min_batch_aligned_elements_B == 0;
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for XE 2D copy.\n");
    }

    return implementable;
  }

  // Helper functions to select packing for conversion
  template <class SrcType,
            class DstType,
            int Cosize>
  struct select_packing { // Naive packing policy
    static constexpr auto value() {
      return Int<cute::gcd(Cosize, 32 / cute::min(sizeof_bits_v<SrcType>, sizeof_bits_v<DstType>))>{};
    }
  };

  template <class EngineIn,
            class EngineOut, 
            class EngineScales, 
            class EngineZeros, 
            class LayoutIn,
            class LayoutOut,
            class LayoutScales,
            class LayoutZeros,
            class... Ts>
  CUTLASS_DEVICE typename std::enable_if_t<sizeof_bits_v<typename EngineIn::value_type> == 4>
  transform_quant(
    Tensor<EngineIn, LayoutIn> const& in,
    Tensor<EngineOut, LayoutOut>& out,
    Tensor<EngineScales, LayoutScales>& tCrS_input,
    Tensor<EngineZeros, LayoutZeros> tCrZ_input
  ) {
    // TODO (Codeplay): add assert here because int4 is not currently supported
    static_assert(!IsATransformed);

    static_assert(is_rmem<EngineIn>::value, "Input tensor for conversion must come from registers");
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
    static_assert(std::is_same_v<typename EngineOut::value_type, typename EngineScales::value_type>);

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;
    using ZeroType = typename EngineZeros::value_type;
    using ScaleType = typename EngineScales::value_type;

    static constexpr auto DPAS = decltype(size<0>(in))::value;
    static constexpr auto N = decltype(size<1>(in))::value;
    static constexpr auto K = decltype(size<2>(in))::value;

    using format_type = ushort;
    static constexpr auto src_bits = sizeof_bits_v<SrcType>;
    static constexpr auto scalar = sizeof_bits_v<format_type> / src_bits;
    static constexpr auto loop_cnt = decltype(size(out))::value / N;
    static_assert((scalar % N) == 0);

    // for tuning performance
    static constexpr auto vec_size = scalar;
    static constexpr auto splits = loop_cnt / vec_size;
    static_assert(vec_size <= scalar);

    // reshape tensors for easy access
    auto s_tensor = make_tensor((format_type*)(raw_pointer_cast(in.data())), Shape<Int<loop_cnt / scalar>, Int<N>>{});
    auto d_tensor = make_tensor(out.data(), Shape<Int<vec_size>, Int<splits>, Int<N>>{});

    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < N; n++) {
      const auto ts = tCrS_input(n);
      const auto tz = [&](){
        if constexpr (sizeof_bits_v<ZeroType> >= 8) {
          return tCrZ_input(n);
        } else {
          return tCrZ_input(n).get();
        }
      }();

      auto& src = *(cute::array<format_type, loop_cnt / scalar>*)(s_tensor(_, n).data());

      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < splits; s++) {
        auto idx =  vec_size * s / scalar;
        auto format_data = src[idx];

        auto& dst = *(cute::array<DstType, vec_size>*)(d_tensor(_, s, n).data());

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < vec_size; i++) {
          auto data = [&](){
            if constexpr (cutlass::platform::numeric_limits<SrcType>::is_signed) {
              return static_cast<SrcType>((format_data >> (src_bits * i)) & 0xf);
            } else {
              return (format_data >> (src_bits * i)) & 0xf;
            }
          }();

          if constexpr (ModeHasScales) {
            if constexpr (IsATransformed) {
              static_assert(dependent_false<LayoutIn> && "ATransform not support now");
            } else {
              using ret_type = cute::conditional_t<sizeof_bits_v<ZeroType> >= 8, ZeroType, int8_t>;
              ret_type minus(data);
              if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
                minus = static_cast<ret_type>(data) - static_cast<ret_type>(tz);
              }
              dst[i] = (static_cast<ScaleType>(minus)) * ts;
            }
          } else {
            dst[i] = static_cast<DstType>(data);
          }
        }
      }
    }
  }

  /// Utilities to transform A.
  template <class EngineIn,
            class EngineOut, 
            class EngineScales, 
            class EngineZeros, 
            class LayoutIn,
            class LayoutOut,
            class LayoutScales,
            class LayoutZeros,
            class... Ts>
  CUTLASS_DEVICE typename std::enable_if_t<sizeof_bits_v<typename EngineIn::value_type> >= 8>
  transform_quant(
    Tensor<EngineIn, LayoutIn> const& tCrA_load, 
    Tensor<EngineOut, LayoutOut>& tCrA_mma,
    Tensor<EngineScales, LayoutScales>& tCrS_input,
    Tensor<EngineZeros, LayoutZeros>& tCrZ_input
  ) {

    static_assert(is_rmem<EngineIn>::value, "Input tensor for A conversion must come from registers");
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
    static_assert(std::is_same_v<typename EngineOut::value_type, typename EngineScales::value_type>);
    static_assert(std::is_same_v<LayoutScales, LayoutZeros>);

    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    if constexpr(cute::is_any_of_v<ElementA,bfloat16_t,half_t,float_e4m3_t,float_e5m2_t>
              && cute::is_any_of_v<ElementB,bfloat16_t,half_t,float_e4m3_t,float_e5m2_t>) {
      convert_FP8_to_FP16<ElementQuant>(make_tensor(reinterpret_cast<const uint8_t*>(tCrA_load.data()), tCrA_load.layout()), tCrA_mma);
    } else {
      auto const& src = tCrA_load(_, _, _);
      auto const& dst = tCrA_mma(_, _, _);
      auto pSrc = const_cast<SrcType*>(raw_pointer_cast(src.data()));
      auto pDst = const_cast<DstType*>(raw_pointer_cast(dst.data()));
      constexpr int num_elements = decltype(size(src))::value;

    // TODO(Codeplay): (perf) consider replacing `pack` with `num_elements` here - See xe_flash_attn_mma.hpp
      constexpr int pack = decltype(select_packing<SrcType, DstType, num_elements>::value())::value;
      using Converter = cutlass::NumericArrayConverter<DstType, SrcType, pack, cutlass::FloatRoundStyle::round_to_nearest>;
      using SrcArray = cutlass::Array<SrcType, pack>;
      using DstArray = cutlass::Array<DstType, pack>;
      constexpr int iters = num_elements / pack;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < iters; ++i) {
        SrcArray const* pSrcArr = reinterpret_cast<SrcArray const*>(pSrc) + i;
        DstArray* pDstArr = reinterpret_cast<DstArray*>(pDst) + i;
        *pDstArr = Converter::convert(*pSrcArr);
      }
    }

    if constexpr (ModeHasScales) {
      if constexpr(IsATransformed){
        // The current scale load atom (1x32) gives 2 scale values to
        // each thread. All threads need access to all other threads
        // scale values, and each scale value is reused twice (unrolled)
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 16; ++i) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 2; ++j) {
            auto scale = shfl_sync(0xFFFFFFFF, tCrS_input(j), i);
            if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero){
              auto zero = shfl_sync(0xFFFFFFFF, tCrZ_input(j), i);
              tCrA_mma(_, _, 0)[j * 16 + i] -= zero;
              tCrA_mma(_, _, 1)[j * 16 + i] -= zero;
            }
            tCrA_mma(_, _, 0)[j * 16 + i] *= scale;
            tCrA_mma(_, _, 1)[j * 16 + i] *= scale;
          }
        }
      } else {
        static constexpr auto N = decltype(size<1>(tCrA_load))::value;

        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < N; ++n) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < decltype(size(tCrA_load))::value / N; ++i) {
            if constexpr (KernelConversionMode == ConversionMode::ConvertAndScaleWithZero){
              tCrA_mma(_, n, _)[i] -= tCrZ_input(n);
            }
            tCrA_mma(_, n, _)[i] *= tCrS_input(n);
          }
        }
      }
    }
  }

template <typename LoadTensors>
CUTLASS_DEVICE auto create_copies(LoadTensors const& load_tensors) {
    Copy_A tiled_copy_a{Copy_A{}.with(get<0>(load_tensors))};
    Copy_B tiled_copy_b{Copy_B{}.with(get<1>(load_tensors))};

    if constexpr(KernelConversionMode == ConversionMode::DirectConvert){
      return cute::make_tuple(tiled_copy_a, tiled_copy_b, Copy_Scale{}, Copy_Zero{});
    }

    Copy_Scale tiled_copy_scale{Copy_Scale{}.with(get<2>(load_tensors))};

    if constexpr(KernelConversionMode == ConversionMode::ConvertAndScale){
      return cute::make_tuple(tiled_copy_a, tiled_copy_b, tiled_copy_scale, Copy_Zero{});
    }

    Copy_Zero tiled_copy_zero{Copy_Zero{}.with(get<3>(load_tensors))};
    return cute::make_tuple(tiled_copy_a, tiled_copy_b, tiled_copy_scale, tiled_copy_zero);
}

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class BlkCoord,
    class LoadTensors
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      BlkCoord const &blk_coord,
      int const &K_start,
      int thread_idx,
      Params const& mainloop,
      LoadTensors const& load_tensors) 
  {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    auto [tiled_copy_a, tiled_copy_b, tiled_copy_scale, tiled_copy_zero] = create_copies(load_tensors);

    // Partition the copying of A and B tiles across the threads
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);
    auto thr_copy_scale = tiled_copy_scale.get_slice(thread_idx);
    auto thr_copy_zero = tiled_copy_zero.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    auto sg = compat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * DispatchPolicy::SubgroupSize;
    auto thr_mma = tiled_mma.get_slice(first_thread_in_sg_idx);

    // Partition
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);

    // Create fragments
    Tensor mma_A = make_tensor<ElementMMA>(make_fragment_layout(tiled_copy_a, tCgA(_,_,_,0).shape()));
    Tensor mma_B = make_tensor<ElementMMA>(make_fragment_layout(tiled_copy_b, tCgB(_,_,_,0).shape()));

    // If IsATransformed, we need modes M_atom, and M_iter from fragment_A
    // layout else we need mode N_iter from fragment_B layout.
    static constexpr auto scale_traits_size = decltype(size(typename GmemTiledCopyScale::BlockShape{}))::value / SubgroupSize;
    static constexpr auto scale_traits_num = SG_N / size<1>(typename GmemTiledCopyScale::BlockShape{});
    using FragScaleLayout = std::conditional_t<IsATransformed,
                                               Layout<Shape<_2, _1, _1>>,
                                               Layout<Shape<Int<scale_traits_size>, Int<scale_traits_num>, _1>>>;
    Tensor fragment_scale_input = make_tensor<NonVoidElementScale>(FragScaleLayout{});

    static constexpr auto zero_traits_size = decltype(size(typename GmemTiledCopyZero::BlockShape{}))::value / SubgroupSize;
    static constexpr auto zero_traits_num = SG_N * zero_elements_packed_along_k / size<1>(typename GmemTiledCopyZero::BlockShape{});
    using FragZeroLayout = std::conditional_t<IsATransformed,
                                               Layout<Shape<_2, _1, _1>>,
                                               Layout<Shape<Int<zero_traits_size>, Int<zero_traits_num>, _1>>>;
    Tensor fragment_zero_input =  make_tensor<NonVoidElementZero> (FragZeroLayout{});

    // narrow input fragment
    Tensor quant_frag = make_tensor<ElementQuant>(
        std::conditional_t<IsATransformed, decltype(mma_A.layout()),
                           decltype(mma_B.layout())>{});

    static_assert(std::is_same_v<typename decltype(quant_frag)::value_type, ElementQuant>);
    static_assert(std::is_same_v<typename decltype(mma_A)::value_type, ElementMMA>);
    static_assert(std::is_same_v<typename decltype(mma_B)::value_type, ElementMMA>);

    // Retile for copy
    auto [frag_copy_A, frag_copy_B] = [&](){
      if constexpr (IsATransformed) {
        return std::make_pair(thr_copy_A.retile_D(quant_frag), thr_copy_B.retile_D(mma_B));
      } else {
        return std::make_pair(thr_copy_A.retile_D(mma_A), thr_copy_B.retile_D(quant_frag));
      }
    }();

    Tensor copy_tCrS = thr_copy_scale.retile_D(fragment_scale_input);
    Tensor copy_tCrZ = thr_copy_zero.retile_D(fragment_zero_input);

    // Retile global tile for copies
    Tensor tAgA = thr_copy_A.retile_S(tCgA);
    Tensor tBgB = thr_copy_B.retile_S(tCgB);

    auto tiled_prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(tiled_copy_a);;
    auto tiled_prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(tiled_copy_b);;
    auto thr_prefetch_A = tiled_prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = tiled_prefetch_b.get_slice(thread_idx);

    // Partition global tile for prefetch
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    //
    // Mainloop
    //
    // TODO(Codeplay): Define these coord tensors using proper cute logic 
    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
    const int l_coord = 0;

    Tensor copy_iter_s = [&](){
      if constexpr(IsATransformed){
        return make_tensor(make_inttuple_iter(make_coord(m_coord, 0, l_coord)),
                           make_layout(make_shape(_2{}, _1{}, _1{}, k_tile_count), 
                                       make_stride(E<0>{} * _16{}, E<0>{} * _32{}, _0{}, E<1>{} * _1{})));
      }else{
        return make_tensor(make_inttuple_iter(make_coord(n_coord, 0, l_coord)),
                           make_layout(make_shape(Int<scale_traits_size>{}, Int<scale_traits_num>{}, _1{}, k_tile_count),
                                       make_stride(E<0>{} * _16{}, E<0>{} * size<1>(typename GmemTiledCopyScale::BlockShape{}), _0{}, E<1>{} * _1{})));
      }
    }();

    Tensor copy_iter_z = [&](){
      if constexpr(IsATransformed){
        return make_tensor(make_inttuple_iter(make_coord(m_coord, 0, l_coord)),
                           make_layout(make_shape(_2{}, _1{}, _1{}, k_tile_count),
                                       make_stride(E<0>{} * _16{}, E<0>{} * _32{}, _0{}, E<1>{} * _1{})));
      }else{
        return make_tensor(make_inttuple_iter(make_coord(n_coord * zero_elements_packed_along_k, 0, l_coord)),
                           make_layout(make_shape(Int<zero_traits_size>{}, Int<zero_traits_num>{}, _1{}, k_tile_count),
                                       make_stride(E<0>{} * _16{}, E<0>{} * size<1>(typename GmemTiledCopyZero::BlockShape{}), _0{}, E<1>{} * _1{})));
      }
    }();

  #if CUTLASS_ENABLE_DEBUG_PRINTS
  #define PRINT(x) print(#x ": "); print(x); print("\n");
    if (cutlass::thread(LOG_THREAD, LOG_GROUP)) {
        print("======================= A: \n");
        PRINT(gA);  
        PRINT(tCgA);
        PRINT(tAgA);
        PRINT(mma_A);
        PRINT(frag_copy_A);

        print("=====================  B :\n");
        PRINT(gB);
        PRINT(tCgB);
        PRINT(tBgB);
        PRINT(mma_B);
        PRINT(frag_copy_B);

        print("=====================  Config: \n");
        PRINT(MaxThreadsPerBlock);
        PRINT(SubgroupTileShape{});

        PRINT(tiled_prefetch_a);
        PRINT(tiled_prefetch_b);
        PRINT(pAgA);
        PRINT(pBgB);
      }
  #undef PRINT
  #endif

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    constexpr int barrier_scope = 2;
    int prefetch_k = k_start_idx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
      prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
    }

    const int k_reload_factor = mainloop.group_size / BLK_K; 

    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);

      // Copy gmem to rmem for the first k_tile
      copy(tiled_copy_a, tAgA(_,_,_,k_tile), frag_copy_A);
      copy(tiled_copy_b, tBgB(_,_,_,k_tile), frag_copy_B);

      if constexpr(ModeHasScales) {
        copy(tiled_copy_scale, copy_iter_s(_, _, _, k_tile / k_reload_factor), copy_tCrS);
      }
      if constexpr(KernelConversionMode == ConversionMode::ConvertAndScaleWithZero) {
        copy(tiled_copy_zero, copy_iter_z(_, _, _, k_tile / k_reload_factor / zero_elements_packed_along_k), copy_tCrZ);
      }

      if(prefetch_k < k_tile_count) {
        prefetch(tiled_prefetch_a, pAgA(_,_,_,prefetch_k));
        prefetch(tiled_prefetch_b, pBgB(_,_,_,prefetch_k));
      }

      if constexpr (IsATransformed) {
        transform_quant(quant_frag, mma_A, fragment_scale_input,
                        fragment_zero_input);
      } else {
        if constexpr (ModeHasScalesZero && sizeof_bits_v<NonVoidElementZero> < 8) {
          transform_quant(quant_frag, mma_B, fragment_scale_input, fragment_zero_input((k_tile / k_reload_factor) % zero_traits_size, _, 0));
        } else {
          transform_quant(quant_frag, mma_B, fragment_scale_input, fragment_zero_input);
        }
      }

      cute::gemm(tiled_mma, mma_A, mma_B, accum);
      barrier_wait(barrier_scope);
    }
  }

  template <typename ProblemShape_MNKL>
  CUTLASS_DEVICE MainloopTensors update_tensor_shape_stride(
    Params const& mainloop_params,
    int32_t const& next_group,
    ProblemShape_MNKL const& problem_shape_mnkl) {
    const int32_t M = get<0>(problem_shape_mnkl);
    const int32_t N = get<1>(problem_shape_mnkl);
    const int32_t K = get<2>(problem_shape_mnkl);

    ElementA const* ptr_A_curr_batch = reinterpret_cast<ElementA const*>(mainloop_params.ptr_A[next_group]);
    auto ptr_B_curr_batch = [&]() {
      if constexpr (sizeof_bits_v<ElementB> < 8) {
        return cute::subbyte_iterator<const ElementB>(mainloop_params.ptr_B[next_group]);
      } else {
        return make_gmem_ptr(static_cast<ElementB const *>(mainloop_params.ptr_B[next_group]));
      }
    }();

    TensorMKL mA_mkl = make_tensor(make_gmem_ptr(ptr_A_curr_batch), make_shape(M, K, static_cast<int32_t>(1)), mainloop_params.dA[next_group]);
    TensorNKL mB_nkl = make_tensor(ptr_B_curr_batch, make_shape(N, K,static_cast<int32_t>(1)), mainloop_params.dB[next_group]);

    if constexpr(KernelConversionMode == ConversionMode::DirectConvert){
      return cute::make_tuple(mA_mkl, mB_nkl, TensorS{}, TensorZ{});
    }

    auto scale_k = cute::ceil_div(K, mainloop_params.group_size);
    TensorS mScale = make_tensor(
        make_gmem_ptr(static_cast<NonVoidElementScale const *>(mainloop_params.ptr_S[next_group])),
        make_layout(make_shape(IsATransformed ? M : N, scale_k, static_cast<int32_t>(1)), mainloop_params.dS[next_group]));

    if constexpr(KernelConversionMode == ConversionMode::ConvertAndScale){
      return cute::make_tuple(mA_mkl, mB_nkl, mScale, TensorZ{});
    }

    auto ptr_Z = [&]() {
      if constexpr (sizeof_bits_v<NonVoidElementZero> < 8) {
        return cute::subbyte_iterator<const NonVoidElementZero>(mainloop_params.ptr_Z[next_group]);
      } else {
        return make_gmem_ptr(static_cast<NonVoidElementZero const *>(mainloop_params.ptr_Z[next_group]));
      }
    }();

    TensorZ mZero = make_tensor(ptr_Z,
                    make_layout(make_shape(zero_elements_packed_along_k * (IsATransformed ? M : N), scale_k / zero_elements_packed_along_k, static_cast<int32_t>(1)),
                    make_stride(_1{}, static_cast<int64_t>(zero_elements_packed_along_k) * (IsATransformed ? M : N), static_cast<int64_t>(IsATransformed ? M : N) * scale_k)));

    return cute::make_tuple(mA_mkl, mB_nkl, mScale, mZero);
  }
};


} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
