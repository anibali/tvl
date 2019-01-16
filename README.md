Building extensions:

```bash
rm -rf build && python setup.py build_ext --inplace && cp ext/tvlnv.py src/ && pytest -s
```
