'use strict';

const jimp = require("jimp");
const assert = require('assert');
const fs = require('fs');
const _ = require('lodash');
const menoh = require('..'); // This menoh module

const CONV1_1_IN_NAME = "140326425860192";
const FC6_OUT_NAME = "140326200777584";
const SOFTMAX_OUT_NAME = "140326200803680";

const SYNSET_WORDS_PATH = '../test/data/vgg16/synset_words.txt';
const INPUT_IMAGE_LIST = [
    '../test/data/vgg16/Light_sussex_hen.jpg',
    '../test/data/vgg16/honda_nsx.jpg'
];

// Crop the original image to square shape.
function cropToSquare(image) {
    let x = 0, y = 0;
    let w = image.bitmap.width;
    let h = image.bitmap.height;
    if (w < h) {
        const dH = h - w;
        h = w;
        x = Math.floor(dH / 2);
    } else if (w > h) {
        const dW = w - h;
        w = h;
        x = Math.floor(dW / 2);
    }
    image.crop(x, y, w, h);
}

// Load all image files.
function loadInputImages() {
    return Promise.all(INPUT_IMAGE_LIST.map((filename) => jimp.read(filename)));
}

// Load and create a category list from file.
function loadCategoryList() {
    const text = fs.readFileSync(SYNSET_WORDS_PATH, 'utf8');
    return text.split('\n').map((line) => line.trim());
}

// Find the indexes of the k largest values.
function findIndicesOfTopK(a, k) {
    var outp = [];
    for (var i = 0; i < a.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > k) {
            outp.sort((l, r) => { return a[r] - a[l]; });
            outp.pop();
        }
    }
    return outp;
}

console.log('Using menoh core version %s', menoh.getNativeVersion());

const categoryList = loadCategoryList();

loadInputImages()
.then((imageList) => {
    const data = [];
    imageList.forEach((image, batchIdx) => {
        // Crop the input image to a square shape.
        cropToSquare(image);

        // Resize it to 224 x 224.
        image.resize(224, 224);

        // Convert bitmap to an array.
        const numPixels = image.bitmap.width * image.bitmap.height;
        const batchOffset = batchIdx * numPixels * 3;
        image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
            for (let c = 0; c < 3; ++c) {
                const dataIdx = c * numPixels + y * image.bitmap.width + x + batchOffset;
                data[dataIdx] = this.bitmap.data[idx + c];
            }
        });
    });

    // Load ONNX file
    return menoh.create('../test/data/vgg16/VGG16.onnx')
    .then((builder) => {
        const batchSize = imageList.length;

        // Add input
        builder.addInput(CONV1_1_IN_NAME, [
            batchSize,  // 2 images in the data
            3,          // number of channels
            224,        // height
            224         // width
        ]);

        // Add output
        builder.addOutput(FC6_OUT_NAME);
        builder.addOutput(SOFTMAX_OUT_NAME);

        // Build a new Model
        const model = builder.buildModel({
            backendName: 'mkldnn'
        })

        // Set input data
        model.setInputData(CONV1_1_IN_NAME, data);

        // Run the model
        return model.run()
        .then(() => {
            const out1 = model.getOutput(FC6_OUT_NAME);
            const out2 = model.getOutput(SOFTMAX_OUT_NAME);

            // just to be sure
            assert.equal(out1.dims[0] * out1.dims[1], out1.data.length);
            assert.equal(out2.dims[0] * out2.dims[1], out2.data.length);
            assert.equal(out1.dims[0], batchSize); // only applies to this example
            assert.equal(out2.dims[0], batchSize); // only applies to this example

            // Print the results.
            out1.data = _.chunk(out1.data, out1.dims[1]); // reshaped
            out2.data = _.chunk(out2.data, out2.dims[1]); // reshaped
            
            for (let bi = 0; bi < batchSize; ++bi) {
                console.log('### Result for %s', INPUT_IMAGE_LIST[bi]);
                console.log('fc6 out: %s ...', out1.data[bi].slice(0, 5).join(' '));

                const topK = findIndicesOfTopK(out2.data[bi], 5);
                console.log('Top 5 categories are:');
                topK.forEach((i) => {
                    console.log('[%d] %f %s', i, out2.data[bi][i], categoryList[i]);
                });
            }

            // Happily done!
        });
    });
});

