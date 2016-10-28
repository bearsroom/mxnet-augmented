/*
 * MXNET implementation of center loss
 * Ref: A Discriminative Feature Learning Approach for Deep Face Recognition
 */


#include "./center_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateCenterLossOp<gpu>(CenterLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CenterLossOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

