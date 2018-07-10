
#include <nan.h>
#include "model.h"

namespace nodeMenoh {

NAN_MODULE_INIT(InitAll) {
    ModelBuilder::Init(target);
    Model::Init(target);
}

NODE_MODULE(NODE_GYP_MODULE_NAME, InitAll)

}  // namespace nodeMenoh
