#!/bin/bash
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo $SITE_PACKAGES

cp /app/lib/worker.py $SITE_PACKAGES/torch/utils/data/_utils/
cp /app/lib/fetch.py $SITE_PACKAGES/torch/utils/data/_utils/
# cp /app/lib/sampler.py $SITE_PACKAGES/torch/utils/data/
# cp /app/lib/__init__.py $SITE_PACKAGES/torch/utils/data/