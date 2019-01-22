Building extensions:

```bash
rm -rf build && python setup.py build_ext --inplace && pytest -s
```

Dockerised build and run tests:

```bash
docker build -t tvl . && docker run --rm -it tvl
```

## TODO

* Some of the stuff in the TvlnvFrameReader constructor should probably be moved to
  a global init function of some variety
* Make specific backends optional
