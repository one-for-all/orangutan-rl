const { rspack } = require("@rspack/core");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");
const path = require("path");

const isDev = process.env.NODE_ENV === "development";
const dist = path.resolve(__dirname, "../docs");

module.exports = (env) => {
  return {
    mode: isDev ? "development" : "production",
    entry: "./src/index.ts",
    devtool: isDev ? "inline-source-map" : false,
    output: {
      path: dist,
      filename: "bundle.js",
      clean: true,
    },
    resolve: {
      extensions: [".ts", ".js"],
      alias: {
        three: path.resolve(__dirname, "node_modules/three"),
      },
    },
    experiments: {
      asyncWebAssembly: true,
      syncWebAssembly: true,
    },
    module: {
      rules: [
        {
          test: /\.ts$/,
          loader: "builtin:swc-loader",
        },
      ],
    },
    plugins: [
      new rspack.CopyRspackPlugin({
        patterns: [{ from: "static", to: dist }],
      }),

      new WasmPackPlugin({
        crateDirectory: "../",
        watchDirectories: [
          path.resolve(__dirname, "../../gorilla-physics/src"),
          path.resolve(__dirname, "static"),
        ],
      }),
    ],
    // To disable warning on screen
    stats: {
      warnings: false,
    },
    performance: {
      hints: false,
    },
    cache: {
      // for speeding up the rebuild
      type: "filesystem",
      buildDependencies: {
        config: [__filename],
      },
    },
  };
};
