/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cu
 * \brief
 * \author Bing Xu
*/

#include "./softmax_with_neg_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(SoftmaxWithNegativeOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxWithNegativeOutputOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

