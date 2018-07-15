'use strict';

const jimp = require("jimp");
const assert = require('assert');
const fs = require('fs');
const ndarray = require('ndarray');
const dtype = require('dtype');
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
    for (var i = 0; i < a.size; i++) {
        outp.push(i); // add index to output array
        if (outp.length > k) {
            outp.sort((l, r) => { return a.get(r) - a.get(l); });
            outp.pop();
        }
    }
    return outp;
}

console.log('Using menoh core version %s', menoh.getNativeVersion());

const categoryList = loadCategoryList();

// Load ONNX file
return menoh.create('../test/data/vgg16/VGG16.onnx')
.then((builder) => {
    const batchSize = INPUT_IMAGE_LIST.length;

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

    // Create a view for input buffer using ndarray.
    const iData = (function () {
        const prof = model.getProfile(CONV1_1_IN_NAME);
        return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
    })();

    // Create a view for each output buffers using ndarray.
    const oDataFc6 = (function () {
        const prof = model.getProfile(FC6_OUT_NAME);
        return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
    })();
    const oDataSmx = (function () {
        const prof = model.getProfile(SOFTMAX_OUT_NAME);
        return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
    })();

    return loadInputImages()
    .then((imageList) => {
        const data = [];
        imageList.forEach((image, batchIdx) => {
            // Crop the input image to a square shape.
            cropToSquare(image);

            // Resize it to 224 x 224.
            image.resize(224, 224);

            // Now, copy the image data into to the input buffer in NCHW format.
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                for (let c = 0; c < 3; ++c) {
                    const val = image.bitmap.data[idx + c];
                    iData.set(batchIdx, c, y, x, val);
                }
            });
        });

        // Run the model
        return model.run()
        .then(() => {
            // Print the results.
            for (let bi = 0; bi < batchSize; ++bi) {
                console.log('### Result for %s', INPUT_IMAGE_LIST[bi]);
                const fc6 = oDataFc6.pick(bi, null);
                console.log('fc6 out: %s ...', [0, 1, 2].map((i) => fc6.get(i)).join(' '));

                const topK = findIndicesOfTopK(oDataSmx.pick(bi, null), 5);
                console.log('Top 5 categories are:');
                topK.forEach((i) => {
                    console.log('[%d] %f %s', i, oDataSmx.get(bi, i), categoryList[i]);
                });
            }
        });
    });
})
.catch((err) => {
    console.log('Error:', err);
});

