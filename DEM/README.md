# Star Discrepancy

Algorithms for the star discrepancy. Based on the original
implementations by Magnus Wahlström, made available in the `original`
folder with his permission for testing and benchmarking purposes.

### Compiling

To compile the code use:

```
cmake -S . -B build
cmake --build build
```

By default this builds the project in `Release` mode.

### Development

For development purposes you should configure the project with the
development flag turned on:

```
cmake -S . -B build -DSTAR_DISCREPANCY_DEVELOPMENT_BUILD=ON
cmake --build build
```

If you also want to configure it in `Debug` mode do:

```
cmake -S . -B build -DSTAR_DISCREPANCY_DEVELOPMENT_BUILD=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

##### Testing

To run the tests use

```
cmake --build build --target tests
```

##### Benchmarking

To run the DEM algorithm benchmarks (ideally in Release mode) run:

```
cmake --build build
cd bench
bash bench_dem.bash
```
