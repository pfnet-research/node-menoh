
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <menoh/version.h>
#include "model.h"

namespace nodeMenoh {

////////////////////////////////////////////////////////////////////////////////
// ModelBuilder class

Nan::Persistent<v8::Function> ModelBuilder::constructor;

ModelBuilder::ModelBuilder() :  _data(NULL),
                                _vptBuilder(NULL),
                                _vpt(NULL),
                                _ivNames() {
}

ModelBuilder::~ModelBuilder() {
    if (_vpt) {
        menoh_delete_variable_profile_table(_vpt);
    }

    if (_vptBuilder) {
        menoh_delete_variable_profile_table_builder(_vptBuilder);
    }

    if (_data) {
        menoh_delete_model_data(_data);
    }
}


NAN_MODULE_INIT(ModelBuilder::Init) {
    target->Set(Nan::New("create").ToLocalChecked(),
                Nan::New<v8::FunctionTemplate>(Create)->GetFunction());
    target->Set(Nan::New("getNativeVersion").ToLocalChecked(),
                Nan::New<v8::FunctionTemplate>(GetNativeVersion)->GetFunction());

    v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
    tpl->SetClassName(Nan::New("ModelBuilder").ToLocalChecked());
    tpl->InstanceTemplate()->SetInternalFieldCount(1);
    Nan::SetPrototypeMethod(tpl, "addInput", AddInput);
    Nan::SetPrototypeMethod(tpl, "addOutput", AddOutput);
    Nan::SetPrototypeMethod(tpl, "buildModel", BuildModel);
    constructor.Reset(tpl->GetFunction());
    target->Set(Nan::New("ModelBuilder").ToLocalChecked(), Nan::GetFunction(tpl).ToLocalChecked());
}

NAN_METHOD(ModelBuilder::New) {
    if (info.IsConstructCall()) {
        // Invoked as constructor: `new ModelBuilder(...)`
        ModelBuilder* mb = new ModelBuilder();
        mb->Wrap(info.This());
        info.GetReturnValue().Set(info.This());

    } else {
        // Invoked as plain function `ModelBuilder(...)`, turn into construct call.
        const int argc = 0;
        v8::Local<v8::Function> cons = Nan::New<v8::Function>(constructor);
        info.GetReturnValue().Set(Nan::NewInstance(cons, argc, NULL).ToLocalChecked());
    }
}

NAN_METHOD(ModelBuilder::Create) {
    if (info.Length() < 2) {
        // Throw an Error that is passed back to JavaScript
        Nan::ThrowTypeError("node-menoh insufficient number of arguments");
        return;
    }
    // Check the argument types
    if (!info[0]->IsString()) {
        Nan::ThrowTypeError("node-menoh arg 1 must be a string");
        return;
    }
    // Check the argument types
    if (!info[1]->IsFunction()) {
        Nan::ThrowTypeError("node-memoh arg 2 must be a function");
        return;
    }

    v8::String::Utf8Value onnxPath(info[0]);
    std::string path(*onnxPath, onnxPath.length());

    // This cb will be deleted by AsyncWorker::~AsyncWorker().
    Nan::Callback *cb = new Nan::Callback(info[1].As<v8::Function>());

    // Start import worker.
    LoadWorker *w = new LoadWorker(cb, path);
    Nan::AsyncQueueWorker(w);

    info.GetReturnValue().Set(Nan::Undefined());
}


NAN_METHOD(ModelBuilder::GetNativeVersion) {
    info.GetReturnValue().Set(Nan::New(MENOH_VERSION_STRING).ToLocalChecked());
}


NAN_METHOD(ModelBuilder::AddInput) {
    if (info.Length() < 2) {
        // Throw an Error that is passed back to JavaScript
        Nan::ThrowTypeError("node-menoh insufficient number of arguments");
        return;
    }
    if (!info[0]->IsString()) {
        Nan::ThrowTypeError("node-menoh arg 1 must be a string");
        return;
    }
    if (!info[1]->IsObject()) {
        Nan::ThrowTypeError("node-menoh arg 2 must be an array");
        return;
    }

    v8::Local<v8::Object> _dataObj;
    v8::Local<v8::Array> data;

    ModelBuilder* mb = ObjectWrap::Unwrap<ModelBuilder>(info.Holder());

    // info[0] - name
    v8::String::Utf8Value _name(info[0]);
    std::string name(*_name, _name.length());

    // info[1] - dims
    _dataObj = info[1]->ToObject();
    data = v8::Local<v8::Array>::Cast(_dataObj);
    std::vector<int> dims;
    for (uint32_t i = 0; i < data->Length(); ++i) {
        v8::Local<v8::Value> _it = Nan::Get(data, i).ToLocalChecked();
        int it = _it->Int32Value();
        dims.push_back(it);
    }
    
    menoh_error_code ec;

    if (dims.size() == 2) {
        ec = menoh_variable_profile_table_builder_add_input_profile_dims_2(
            mb->_vptBuilder, name.c_str(), menoh_dtype_float,
            dims[0], dims[1]);
        if (ec) {
            Nan::ThrowTypeError(menoh_get_last_error_message());
            return;
        }
    } else if (dims.size() == 4) {
        ec = menoh_variable_profile_table_builder_add_input_profile_dims_4(
            mb->_vptBuilder, name.c_str(), menoh_dtype_float,
            dims[0], dims[1], dims[2], dims[3]);
        if (ec) {
            Nan::ThrowTypeError(menoh_get_last_error_message());
            return;
        }
    } else {
        Nan::ThrowTypeError("node-menoh size of input dims must be 2 or 4");
        return;
    }

    // Remember the input variable name.
    // This is used later to determine the input buffer size.
    mb->_ivNames.push_back(name);

    info.GetReturnValue().Set(Nan::Undefined());
}

NAN_METHOD(ModelBuilder::AddOutput) {
    if (info.Length() < 1) {
        // Throw an Error that is passed back to JavaScript
        Nan::ThrowTypeError("node-menoh insufficient number of arguments");
        return;
    }
    if (!info[0]->IsString()) {
        Nan::ThrowTypeError("node-menoh arg 1 must be a string");
        return;
    }

    v8::Local<v8::Object> _dataObj;
    v8::Local<v8::Array> data;

    ModelBuilder* mb = ObjectWrap::Unwrap<ModelBuilder>(info.Holder());

    // info[0] - name
    v8::String::Utf8Value _name(info[0]);
    std::string name(*_name, _name.length());

    menoh_error_code ec;
    ec = menoh_variable_profile_table_builder_add_output_profile(
            mb->_vptBuilder, name.c_str(), menoh_dtype_float
            );
    if (ec) {
        Nan::ThrowTypeError(menoh_get_last_error_message());
        return;
    }

    info.GetReturnValue().Set(Nan::Undefined());
}

NAN_METHOD(ModelBuilder::BuildModel) {
    if (info.Length() < 1) {
        // Throw an Error that is passed back to JavaScript
        Nan::ThrowTypeError("node-menoh insufficient number of arguments");
        return;
    }
    // Check the argument types
    if (!info[0]->IsObject()) {
        Nan::ThrowTypeError("node-menoh arg 1 must be an object");
        return;
    }

    menoh_error_code ec;
    ModelBuilder* mb = ObjectWrap::Unwrap<ModelBuilder>(info.Holder());

    // Create vpt if not created yet.
    if (!mb->_vpt) {
        // build variable_profile_table
        ec = menoh_build_variable_profile_table(mb->_vptBuilder, mb->_data, &mb->_vpt);
        if (ec) {
            Nan::ThrowTypeError(menoh_get_last_error_message());
            return;
        }

        // optimize
        ec = menoh_model_data_optimize(mb->_data, mb->_vpt);
        if (ec) {
            Nan::ThrowTypeError(menoh_get_last_error_message());
            return;
        }
    }

    // Create a new Model instance.
    const int argc = 1;
    v8::Local<v8::Value> argv[argc] = { info.Holder() };
    v8::Local<v8::Function> cons = Nan::New<v8::Function>(Model::constructor);
    v8::Local<v8::Object> wrappedModel = Nan::NewInstance(cons, argc, argv).ToLocalChecked();
    info.GetReturnValue().Set(wrappedModel);

    // Set up the model
    Model* model = ObjectWrap::Unwrap<Model>(wrappedModel);

    v8::MaybeLocal<v8::Value> _val;
    v8::Local<v8::Object> config = info[0]->ToObject();
    v8::Local<v8::String> key;

    // backendName
    key = Nan::New("backendName").ToLocalChecked();
    if (Nan::Has(config, key).FromJust()) {
        _val = Nan::Get(config, key);
        if (!_val.IsEmpty()) {
            v8::Local<v8::Value> val = _val.ToLocalChecked();
            v8::String::Utf8Value _name(val);
            std::string name(*_name, _name.length());
            model->_backendName = name;
        }
    }

    // backendConfig
    key = Nan::New("backendConfig").ToLocalChecked();
    if (Nan::Has(config, key).FromJust()) {
        _val = Nan::Get(config, key);
        if (!_val.IsEmpty()) {
            v8::Local<v8::Value> val = _val.ToLocalChecked();
            v8::String::Utf8Value _name(val);
            std::string name(*_name, _name.length());
            model->_backendConfig = name;
        }
    }

    ec = model->setUp(mb);
    if (ec) {
        Nan::ThrowTypeError(menoh_get_last_error_message());
        return;
    }
}

////////////////////////////////////////////////////////////////////////////////
// ModelBuilder::LoadWorker class

ModelBuilder::LoadWorker::LoadWorker(
    Nan::Callback *callback,
    const std::string& onnxPath) :  Nan::AsyncWorker(callback), _onnxPath(onnxPath) {
}

ModelBuilder::LoadWorker::~LoadWorker() {
}

void ModelBuilder::LoadWorker::Execute() {
    // Load ONNX model data
    menoh_error_code ec = menoh_make_model_data_from_onnx(_onnxPath.c_str(), &_data);
    if (ec) {
        SetErrorMessage(menoh_get_last_error_message());
        return;
    }
}

void ModelBuilder::LoadWorker::HandleOKCallback() {
    Nan::HandleScope scope;
    const int argc = 0;
    v8::Local<v8::Function> cons = Nan::New<v8::Function>(constructor);
    v8::Local<v8::Object> obj = Nan::NewInstance(cons, argc, NULL).ToLocalChecked();

    // Copy _data to ModelBuilder#_data.
    ModelBuilder* mb = ObjectWrap::Unwrap<ModelBuilder>(obj);
    mb->_data = _data;

    // Create variable profile table builder.
    menoh_error_code ec;
    ec = menoh_make_variable_profile_table_builder(&mb->_vptBuilder);
    if (ec) {
        v8::Local<v8::Value> _argv[] = {
            v8::Exception::Error(Nan::New(menoh_get_last_error_message()).ToLocalChecked())
        };
        callback->Call(1, _argv);
        return;
    }

    v8::Local<v8::Value> _argv[] = { Nan::Undefined(), obj };
    callback->Call(2, _argv);
}

////////////////////////////////////////////////////////////////////////////////
// Model class

Nan::Persistent<v8::Function> Model::constructor;

Model::Model(ModelBuilder *mb): _backendName("mkldnn"),
                                _backendConfig(""),
                                _native(NULL),
                                _ivNames(mb->_ivNames),
                                _inProgress(false) {
}

Model::~Model() {
    // free all input buffers
    InputVarNames::const_iterator it;
    for (it = _ivNames.begin(); it != _ivNames.end(); ++it) {
        std::string const& name(*it);

        void *buf;
        menoh_error_code ec;
        ec = menoh_model_get_variable_buffer_handle(_native, name.c_str(), &buf);
        if (ec) {
            continue;
        }

        ::free(buf);
    }

    if (_native) {
        menoh_delete_model(_native);
    }
}

void Model::Init(v8::Local<v8::Object> exports) {
    // Prepare constructor template
    v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
    tpl->SetClassName(Nan::New("Model").ToLocalChecked());
    tpl->InstanceTemplate()->SetInternalFieldCount(1);

    // Prototype
    Nan::SetPrototypeMethod(tpl, "setInputData", SetInputData);
    Nan::SetPrototypeMethod(tpl, "run", Run);
    Nan::SetPrototypeMethod(tpl, "getOutput", GetOutput);

    constructor.Reset(tpl->GetFunction());
    exports->Set(Nan::New("Model").ToLocalChecked(), Nan::GetFunction(tpl).ToLocalChecked());
}

menoh_error_code Model::setUp(ModelBuilder const *mb) {
    menoh_model_builder_handle modelBuilder;
    menoh_error_code ec;
    ec = menoh_make_model_builder(mb->_vpt, &modelBuilder);
    if (ec) {
        return ec;
    }

    // create input buffer(s)
    InputVarNames::const_iterator it;
    for (it = _ivNames.begin(); it != _ivNames.end(); ++it) {
        std::string const& name(*it);

        int32_t dimsSize;
        ec = menoh_variable_profile_table_get_dims_size(mb->_vpt, name.c_str(), &dimsSize);
        if (ec) {
            goto exit;
        }

        size_t n = 1;
        for (int32_t i = 0; i < dimsSize; ++i) {
            int32_t d;
            ec = menoh_variable_profile_table_get_dims_at(mb->_vpt, name.c_str(), i, &d);
            if (ec) {
                goto exit;
            }
            n *= (size_t)d;
        }

        void *buf = ::calloc(n, sizeof(float));
        ec = menoh_model_builder_attach_external_buffer(
            modelBuilder, name.c_str(), buf);
        if (ec) {
            goto exit;
        }
    }

    // build model
    ec = menoh_build_model( modelBuilder,
                            mb->_data,
                            _backendName.c_str(),
                            _backendConfig.c_str(),
                            &_native);
exit:
    menoh_delete_model_builder(modelBuilder);
    return ec;
}

NAN_METHOD(Model::New) {
    if (info.Length() < 1) {
        // Throw an Error that is passed back to JavaScript
        Nan::ThrowTypeError("node-menoh insufficient number of arguments");
        return;
    }
    if (!info[0]->IsObject()) {
        Nan::ThrowTypeError("node-menoh arg 1 must be an object");
        return;
    }

    v8::Local<v8::Object> modelData = info[0]->ToObject();

    if (info.IsConstructCall()) {
        // Invoked as constructor: `new Model(...)`
        ModelBuilder* mb = ObjectWrap::Unwrap<ModelBuilder>(modelData);
        Model* model = new Model(mb);
        model->Wrap(info.This());
        info.GetReturnValue().Set(info.This());

    } else {
        // Invoked as plain function `Model(...)`, turn into construct call.
        const int argc = 1;
        v8::Local<v8::Value> argv[argc] = { modelData };
        v8::Local<v8::Function> cons = Nan::New<v8::Function>(constructor);
        info.GetReturnValue().Set(Nan::NewInstance(cons, argc, argv).ToLocalChecked());
    }
}

NAN_METHOD(Model::SetInputData) {
    if (info.Length() < 2) {
        // Throw an Error that is passed back to JavaScript
        Nan::ThrowTypeError("node-menoh insufficient number of arguments");
        return;
    }
    if (!info[0]->IsString()) {
        Nan::ThrowTypeError("node-menoh arg 1 must be a string");
        return;
    }
    if (!info[1]->IsObject()) {
        Nan::ThrowTypeError("node-menoh arg 2 must be an array");
        return;
    }

    v8::Local<v8::Object> _dataObj;
    v8::Local<v8::Array> data;

    Model* model = ObjectWrap::Unwrap<Model>(info.Holder());

    // info[0] - name
    v8::String::Utf8Value _name(info[0]);
    std::string name(*_name, _name.length());

    // retrieve buffer handle
    float *buf;
    menoh_error_code ec;
    ec = menoh_model_get_variable_buffer_handle(
        model->_native, name.c_str(), (void**)&buf);
    if (ec) {
        Nan::ThrowTypeError(menoh_get_last_error_message());
        return;
    }

    // info[1] - data
    _dataObj = info[1]->ToObject();
    data = v8::Local<v8::Array>::Cast(_dataObj);

    // copy data into buf
    for (uint32_t i = 0; i < data->Length(); ++i) {
        v8::Local<v8::Value> _it = Nan::Get(data, i).ToLocalChecked();
        buf[i] = (float)_it->NumberValue();
    }

    info.GetReturnValue().Set(Nan::Undefined());
}

NAN_METHOD(Model::Run) {
    Model* model = ObjectWrap::Unwrap<Model>(info.Holder());

    if (info.Length() < 1) {
        // Throw an Error that is passed back to JavaScript
        Nan::ThrowTypeError("node-menoh insufficient number of arguments");
        return;
    }
    if (!info[0]->IsFunction()) {
        Nan::ThrowTypeError("node-menoh arg 1 must be a function");
        return;
    }

    if (model->_inProgress) {
        Nan::ThrowTypeError("node-menoh previous run is in progress");
        return;
    }

    model->_inProgress = true;

    // Start run worker
    Nan::Callback *cb = new Nan::Callback(info[0].As<v8::Function>());
    RunWorker *w = new RunWorker(cb, model);
    Nan::AsyncQueueWorker(w);

    info.GetReturnValue().Set(Nan::Undefined());
}

NAN_METHOD(Model::GetOutput) {
    if (info.Length() < 1) {
        // Throw an Error that is passed back to JavaScript
        Nan::ThrowTypeError("node-menoh insufficient number of arguments");
        return;
    }
    if (!info[0]->IsString()) {
        Nan::ThrowTypeError("node-menoh arg 1 must be a function");
        return;
    }

    // Read output for the given name.
    v8::String::Utf8Value _name(info[0]);
    std::string name(*_name, _name.length());

    Model* model = ObjectWrap::Unwrap<Model>(info.Holder());

    float *buf;
    menoh_error_code ec;
    ec = menoh_model_get_variable_buffer_handle(
        model->_native, name.c_str(), (void**)&buf);
    if (ec) {
        Nan::ThrowTypeError(menoh_get_last_error_message());
        return;
    }

    int32_t dimsSize;
    ec = menoh_model_get_variable_dims_size(model->_native, name.c_str(), &dimsSize);
    if (ec) {
        Nan::ThrowTypeError(menoh_get_last_error_message());
        return;
    }

    v8::Local<v8::Array> dims = Nan::New<v8::Array>();
    size_t n = 1;
    for (int32_t i = 0; i < dimsSize; ++i) {
        int32_t d;
        ec = menoh_model_get_variable_dims_at(model->_native, name.c_str(), i, &d);
        if (ec) {
            Nan::ThrowTypeError(menoh_get_last_error_message());
            return;
        }
        dims->Set((uint32_t)i, Nan::New(d));
        n *= (size_t)d;
    }

    // Copy whole data into a Javascript array.
    v8::Local<v8::Array> data = Nan::New<v8::Array>();
    for (size_t i = 0; i < n; ++i) {
        float v = buf[i];
        data->Set((uint32_t)i, Nan::New(v));
    }

    // Finally put them in an Javascript object.
    v8::Local<v8::Object> results = Nan::New<v8::Object>();
    results->Set(Nan::New("data").ToLocalChecked(), data);
    results->Set(Nan::New("dims").ToLocalChecked(), dims);

    info.GetReturnValue().Set(results);
}

////////////////////////////////////////////////////////////////////////////////
// Model::RunWorker (inner) class

Model::RunWorker::RunWorker(
    Nan::Callback *callback,
    Model *model) : Nan::AsyncWorker(callback), _model(model) {
}

Model::RunWorker::~RunWorker() {
}

void Model::RunWorker::Execute() {
    menoh_error_code ec = menoh_model_run(_model->_native);
    if (ec) {
        SetErrorMessage(menoh_get_last_error_message());
    }
}

// Called by the main thread.
void Model::RunWorker::HandleOKCallback() {
    _model->_inProgress = false;
    callback->Call(0, NULL); // emit closed event
}

// Called by the main thread.
void Model::RunWorker::HandleErrorCallback() {
    _model->_inProgress = false;
    Nan::AsyncWorker::HandleErrorCallback();
}


}  // namespace nodeMenoh
