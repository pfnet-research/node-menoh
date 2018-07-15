#ifndef NODEMENOH_MODEL_H
#define NODEMENOH_MODEL_H

#include <nan.h>
#include <menoh/menoh.h>

namespace nodeMenoh {

typedef std::vector<std::string> InputVarNames;


class ModelBuilder : public Nan::ObjectWrap {
    public:
        friend class Model;
        class LoadWorker : public Nan::AsyncWorker {
            public:
                friend class ModelBuilder;

            private:
                explicit LoadWorker(Nan::Callback* callback, const std::string& onnxPath);
                virtual ~LoadWorker();

                // Called by the worker thread.
                virtual void Execute();

                // Called by the main therad.
                virtual void HandleOKCallback();

                std::string _onnxPath;
                menoh_model_data_handle _data;
        };

        explicit ModelBuilder();
        ~ModelBuilder();

        static void Init(v8::Local<v8::Object> exports);

    private:
        menoh_model_data_handle _data;
        menoh_variable_profile_table_builder_handle _vptBuilder;
        menoh_variable_profile_table_handle _vpt;
        InputVarNames _ivNames;

        static Nan::Persistent<v8::Function> constructor;
        static NAN_METHOD(New);
        static NAN_METHOD(Create);
        static NAN_METHOD(GetNativeVersion);

        // NodeJS property methods
        static NAN_METHOD(AddInput);
        static NAN_METHOD(AddOutput);
        static NAN_METHOD(BuildModel);
};


class Model : public Nan::ObjectWrap {
    public:
        friend class ModelBuilder;

        class RunWorker : public Nan::AsyncWorker {
            friend class Model;

            public:
                explicit RunWorker(Nan::Callback *callback, Model *model);

            private:
                virtual ~RunWorker();

                // Called by the worker thread.
                void Execute();

                // Called by the main therad.
                virtual void HandleOKCallback();
                virtual void HandleErrorCallback();

                Model *_model;
        };

        static void Init(v8::Local<v8::Object> exports);

        menoh_error_code setUp(ModelBuilder const *mb);

    private:
        explicit Model(ModelBuilder *mb);
        ~Model();

        menoh_error_code getVarInfo(    std::string const& name,
                                        v8::Local<v8::Array>* dims,
                                        size_t *bufSize);

        std::string _backendName;
        std::string _backendConfig;
        menoh_model_handle _native;
        InputVarNames _ivNames;
        bool _inProgress;

        static NAN_METHOD(New);

        // NodeJS property methods
        static NAN_METHOD(SetInputData);
        static NAN_METHOD(Run);
        static NAN_METHOD(GetOutput);
        static NAN_METHOD(GetProfile);

        static Nan::Persistent<v8::Function> constructor;
};

}  // namespace nodeMenoh

#endif//NODEMENOH_MODEL_H
