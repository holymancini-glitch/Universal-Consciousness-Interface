# Troubleshooting Guide - v2.0 Modular Architecture

**Common issues and solutions when using the refactored modules**

---

## 🔍 Quick Diagnosis

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| `ImportError: cannot import name 'X'` | Wrong module path | Try old-style import first |
| `ModuleNotFoundError: No module named 'core.X'` | Missing package | Pull latest code |
| `AttributeError: module has no attribute 'X'` | Typo in class name | Check API reference |
| Different behavior after import | Python cache issue | Clear `__pycache__` |
| Circular import error | Wrong import order | Use module-specific imports |

---

## 🚨 Common Issues

### Issue #1: Cannot Import Class

**Symptom**:
```python
>>> from core.planetary_ecosystem import SomeClass
ImportError: cannot import name 'SomeClass' from 'core.planetary_ecosystem'
```

**Diagnosis**:
- Class name might be misspelled
- Class might be in different module than expected

**Solutions**:

1. **Check what's available**:
```python
import core.planetary_ecosystem as pe
print(pe.__all__)  # Lists all available exports
```

2. **Verify class name**:
```python
# Check if it's actually available
from core import planetary_ecosystem
print(dir(planetary_ecosystem))
```

3. **Try old-style import**:
```python
# Fall back to old import style
from core.planetary_ecosystem_consciousness_network import SomeClass
```

4. **Check the API reference**:
See [`API_REFERENCE_v2.md`](./API_REFERENCE_v2.md) for complete class list

---

### Issue #2: Module Not Found

**Symptom**:
```python
>>> import core.planetary_ecosystem
ModuleNotFoundError: No module named 'core.planetary_ecosystem'
```

**Diagnosis**:
- Code not pulled from branch
- Working in wrong directory
- Python path issue

**Solutions**:

1. **Verify you're on the right branch**:
```bash
git branch
# Should show: claude/analyze-repository-011CUbW7GFeTRTW7Z3weFRi8
```

2. **Pull latest changes**:
```bash
git pull origin claude/analyze-repository-011CUbW7GFeTRTW7Z3weFRi8
```

3. **Check directory exists**:
```bash
ls core/planetary_ecosystem/
# Should show: __init__.py, data_models.py, etc.
```

4. **Verify working directory**:
```python
import os
print(os.getcwd())  # Should be in project root
```

---

### Issue #3: Different Behavior After Switching Imports

**Symptom**:
Code works with old imports but behaves differently with new imports

**Diagnosis**:
- Python bytecode cache issue
- Stale `__pycache__` directories
- Mixed import styles causing confusion

**Solutions**:

1. **Clear Python cache** (Recommended):
```bash
# From project root
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

2. **Force reload modules**:
```python
import importlib
import sys

# Remove from cache
if 'core.planetary_ecosystem' in sys.modules:
    del sys.modules['core.planetary_ecosystem']

# Reimport
from core import planetary_ecosystem
importlib.reload(planetary_ecosystem)
```

3. **Restart Python interpreter**:
```bash
# If using IPython
%reset
# Or just restart your Python session
```

---

### Issue #4: Circular Import Error

**Symptom**:
```python
ImportError: cannot import name 'X' from partially initialized module 'Y'
(most likely due to a circular import)
```

**Diagnosis**:
- Importing from package causes circular dependency
- Module initialization order issue

**Solutions**:

1. **Use module-specific imports**:
```python
# Instead of this (might cause circular import):
from core.planetary_ecosystem import NetworkAnalyzer

# Do this:
from core.planetary_ecosystem.network_analyzer import NetworkAnalyzer
```

2. **Import inside functions** (temporary fix):
```python
def my_function():
    # Import here instead of at module level
    from core.planetary_ecosystem import NetworkAnalyzer
    analyzer = NetworkAnalyzer()
```

3. **Check import order**:
```python
# Good: Import data models first
from core.planetary_ecosystem.data_models import EcosystemType
from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork

# Bad: Importing interdependent modules together
from core.planetary_ecosystem import *  # Can cause issues
```

---

### Issue #5: Missing Dependencies

**Symptom**:
```python
>>> from core.adaptive_learning import AdaptiveLearningSystem
ModuleNotFoundError: No module named 'numpy'
```

**Diagnosis**:
- Required dependencies not installed
- Working in wrong Python environment

**Solutions**:

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Check Python environment**:
```bash
python --version
pip list | grep numpy  # Should show numpy if installed
```

3. **Verify virtual environment**:
```bash
# Activate virtual environment if using one
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

---

### Issue #6: Attribute Error on Package

**Symptom**:
```python
>>> import core.planetary_ecosystem as pe
>>> pe.SomeClass()
AttributeError: module 'core.planetary_ecosystem' has no attribute 'SomeClass'
```

**Diagnosis**:
- Class not exported in `__all__`
- Typo in class name
- Wrong package

**Solutions**:

1. **Check available attributes**:
```python
import core.planetary_ecosystem as pe
print(pe.__all__)  # See what's exported
print([x for x in dir(pe) if not x.startswith('_')])  # See all public attributes
```

2. **Verify class name**:
```python
# Check if name is correct
from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork
# Not: PlanetaryEcosystemNetwork (wrong name)
```

3. **Try direct module import**:
```python
from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork
```

---

### Issue #7: Old Code Stops Working

**Symptom**:
Code that worked before refactoring now fails

**Diagnosis**:
- Should not happen! This indicates a bug
- Might be environment issue

**Solutions**:

1. **Verify backward compatibility**:
```python
# Old import should still work
from core.planetary_ecosystem_consciousness_network import PlanetaryEcosystemConsciousnessNetwork
network = PlanetaryEcosystemConsciousnessNetwork()
```

2. **Run compatibility tests**:
```bash
python test_backward_compatibility.py
```

3. **Check for local modifications**:
```bash
git status  # Should show no unexpected changes
git diff    # Check for uncommitted modifications
```

4. **Report the issue**:
If old imports truly don't work, this is a bug! Please report with:
- The exact import statement that fails
- The full error message
- Your Python version

---

## 🔧 Advanced Troubleshooting

### Debug Import Paths

```python
import sys
import core.planetary_ecosystem

# Check where Python is loading the module from
print(core.planetary_ecosystem.__file__)

# Check Python path
for path in sys.path:
    print(path)
```

### Verify Package Structure

```bash
# Check all files exist
ls core/planetary_ecosystem/
# Should show: __init__.py, data_models.py, network_core.py, etc.

# Check file sizes (facades should be small)
wc -l core/planetary_ecosystem_consciousness_network.py  # Should be ~76 lines
```

### Test Individual Modules

```python
# Test each module independently
import core.planetary_ecosystem.data_models
import core.planetary_ecosystem.network_core
import core.planetary_ecosystem.network_analyzer

# If one fails, that's the problematic module
```

### Check for Syntax Errors

```bash
# Compile all modules to check for syntax errors
python -m py_compile core/planetary_ecosystem/*.py
```

---

## 🐛 Debugging Checklist

When something doesn't work, go through this checklist:

- [ ] **Verified I'm using correct Python version** (3.8+)
- [ ] **Checked I'm on the right git branch**
- [ ] **Pulled latest changes from repository**
- [ ] **Verified package directory exists** (`ls core/planetary_ecosystem/`)
- [ ] **Cleared Python cache** (`find . -name __pycache__ -exec rm -r {} +`)
- [ ] **Restarted Python interpreter**
- [ ] **Checked class name spelling** (API reference)
- [ ] **Tried old-style import as fallback**
- [ ] **Verified dependencies installed** (`pip install -r requirements.txt`)
- [ ] **Checked working directory** (should be project root)
- [ ] **Read error message carefully** (exact wording matters)

---

## 📊 Common Error Messages Decoded

### `ImportError: cannot import name 'X' from 'Y'`
**Meaning**: Python found module Y but class X isn't in it
**Fix**: Check class name spelling or use different import path

### `ModuleNotFoundError: No module named 'X'`
**Meaning**: Python can't find the module at all
**Fix**: Verify directory exists and you're in right location

### `AttributeError: module 'X' has no attribute 'Y'`
**Meaning**: Module loaded but doesn't have that attribute
**Fix**: Check `__all__` to see what's exported

### `ImportError: partially initialized module` (circular import)
**Meaning**: Module imports create a cycle
**Fix**: Use module-specific imports instead of package imports

### `TypeError: 'module' object is not callable`
**Meaning**: Trying to instantiate a module instead of a class
**Fix**: Check you imported the class, not the module

---

## 🎯 Performance Issues

### Slow Imports

**Symptom**: Imports take longer than expected

**Solutions**:
1. Use module-specific imports (faster):
   ```python
   from core.planetary_ecosystem.network_core import PlanetaryEcosystemConsciousnessNetwork
   ```

2. Import only what you need:
   ```python
   # Instead of
   from core.planetary_ecosystem import *

   # Do
   from core.planetary_ecosystem import PlanetaryEcosystemConsciousnessNetwork
   ```

3. Profile imports:
   ```bash
   python -X importtime -c "from core import planetary_ecosystem" 2>&1 | grep planetary
   ```

---

## 💡 Prevention Tips

1. **Always use package imports for new code**
2. **Clear cache when switching branches**
3. **Keep dependencies updated**
4. **Use virtual environments**
5. **Read error messages carefully**
6. **Check API reference when unsure**
7. **Test imports before using in production**

---

## 📞 Getting Help

If you're still stuck:

1. **Check existing documentation**:
   - [`MIGRATION_GUIDE.md`](./MIGRATION_GUIDE.md)
   - [`API_REFERENCE_v2.md`](./API_REFERENCE_v2.md)
   - [`QUICK_REFERENCE.md`](./QUICK_REFERENCE.md)

2. **Run verification tests**:
   ```bash
   python test_backward_compatibility.py
   python test_refactoring_verification.py
   ```

3. **Check test results**:
   - [`REFACTORING_TEST_RESULTS.md`](./REFACTORING_TEST_RESULTS.md)

4. **Ask for help** with:
   - Exact error message (copy-paste)
   - Code that reproduces the issue
   - Python version and OS
   - Which import style you're using

---

## ✅ Success Indicators

You know everything is working when:

- ✅ Old imports work without changes
- ✅ New imports work smoothly
- ✅ No import errors in your code
- ✅ Tests pass successfully
- ✅ Code runs as expected

---

**Remember**: The refactoring maintains 100% backward compatibility. If old imports don't work, that's a bug - please report it!

*Last updated: November 4, 2025*
*Version: 2.0.0*
