#!/bin/bash

# Script para activar el ambiente virtual del proyecto
echo "🚀 Activando ambiente virtual..."

# Cambiar al directorio del proyecto
cd "$(dirname "$0")"

# Activar el ambiente virtual
source .venv/bin/activate

# Verificar instalaciones clave
echo "✅ Verificando instalaciones..."
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)') (CUDA: $(python -c 'import torch; print(torch.cuda.is_available())'))"
echo "   Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "   PyTorch Lightning: $(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)')"
echo "   Pandas: $(python -c 'import pandas; print(pandas.__version__)')"

echo ""
echo "🎯 Ambiente listo para usar!"
echo "   Ejecuta: jupyter notebook para abrir Jupyter"
echo "   O simplemente: python tu_script.py"
echo ""
