'use strict';

let addon;
try {
    addon = require('./build/Release/menoh.node');
} catch (err) {
    console.log('Error:', err);
    addon = require('./build/Debug/menoh.node');
}

// Promisify addon.create()
(function () {
    const create = addon.create;
    addon.create = function (path, cb) {
        if (cb) {
            create.call(this, path, cb);
            return;
        }

        return new Promise((resolve, reject) => {
            create.call(this, path, (err, builder) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve(builder);
            });
        });
    }
})();

// Promisify addon.Model.prototype.run()
(function () {
    const run = addon.Model.prototype.run;
    addon.Model.prototype.run = function (cb) {
        if (cb) {
            run.call(this, cb);
            return;
        }

        return new Promise((resolve, reject) => {
            run.call(this, (err) => {
                if (err) {
                    reject(err);
                    return;
                }
                resolve();
            });
        });
    }
})();

module.exports = addon;

