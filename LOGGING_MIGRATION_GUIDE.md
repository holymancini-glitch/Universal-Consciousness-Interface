# Logging Migration Guide

## Overview

This guide provides instructions for migrating from `print()` statements to proper logging infrastructure across the Universal Consciousness Interface codebase.

## Why Migrate?

**Current Issues with print():**
- No log levels (can't filter by severity)
- No structured logging
- Not suitable for production environments
- Can't be disabled or redirected easily
- No timestamps or module information
- Difficult to aggregate and analyze

**Benefits of Logging:**
- Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Structured output with timestamps and module names
- Can be redirected to files, syslog, or monitoring systems
- Production-ready
- Integrates with monitoring and alerting systems
- Consciousness-specific emoji formatting

## Quick Start

### 1. Import the logging module

```python
# Old way
# (no imports needed for print)

# New way
import logging
from core.logging_config import get_logger

logger = get_logger(__name__)
```

### 2. Replace print statements

```python
# Old way
print("Processing consciousness input...")
print(f"Consciousness level: {level}")
print(f"ERROR: Failed to process: {error}")

# New way
logger.info("Processing consciousness input...")
logger.info(f"Consciousness level: {level}")
logger.error(f"Failed to process: {error}")
```

## Log Levels

Choose the appropriate log level:

### DEBUG (logger.debug)
**Use for:** Detailed diagnostic information, variable dumps, state tracking
```python
logger.debug(f"Internal state: consciousness_vectors={len(vectors)}, cache_hits={hits}")
logger.debug(f"Processing step 3 of 10: vector transformation")
```

### INFO (logger.info)
**Use for:** General informational messages, successful operations, milestones
```python
logger.info("✨ Consciousness system initialized successfully")
logger.info(f"Processing {count} consciousness inputs")
logger.info(f"Consciousness level reached: {level:.3f}")
```

### WARNING (logger.warning)
**Use for:** Something unexpected but not critical, deprecated features, fallbacks
```python
logger.warning("Quantum consciousness module not available, using simulation")
logger.warning(f"Consciousness level {level} below optimal threshold {threshold}")
logger.warning("Memory usage high: {memory_mb:.1f}MB, consider optimization")
```

### ERROR (logger.error)
**Use for:** Error conditions that should be investigated
```python
logger.error(f"Failed to process consciousness input: {error}")
logger.error(f"Integration bridge connection lost: {reason}")
```

### CRITICAL (logger.critical)
**Use for:** System-threatening errors, safety violations, emergency shutdowns
```python
logger.critical("Emergency shutdown triggered: radiation levels unsafe")
logger.critical(f"Consciousness safety violation: {violation_type}")
```

## Migration Patterns

### Pattern 1: Simple Informational Messages

```python
# Before
print("System initialized")

# After
logger.info("System initialized")
```

### Pattern 2: Formatted Output

```python
# Before
print(f"Consciousness level: {level:.3f}, Status: {status}")

# After
logger.info(f"Consciousness level: {level:.3f}, Status: {status}")
```

### Pattern 3: Error Messages

```python
# Before
print(f"ERROR: Failed to connect: {e}")

# After
logger.error(f"Failed to connect: {e}")
```

### Pattern 4: Debug Information

```python
# Before
print(f"Debug: vector_count={len(vectors)}, coherence={coherence}")

# After
logger.debug(f"vector_count={len(vectors)}, coherence={coherence}")
```

### Pattern 5: Multi-line Reports

```python
# Before
print("=" * 60)
print("CONSCIOUSNESS ANALYSIS REPORT")
print("=" * 60)
print(f"Level: {level}")
print(f"Quality: {quality}")

# After - Option 1: Multiple log calls
logger.info("=" * 60)
logger.info("CONSCIOUSNESS ANALYSIS REPORT")
logger.info("=" * 60)
logger.info(f"Level: {level}")
logger.info(f"Quality: {quality}")

# After - Option 2: Single log call with newlines (better for file logs)
report = f"""
{'=' * 60}
CONSCIOUSNESS ANALYSIS REPORT
{'=' * 60}
Level: {level}
Quality: {quality}
"""
logger.info(report)
```

## When to Keep print()

**Keep print() statements in:**
1. `if __name__ == "__main__"` demo/test blocks at the end of modules
2. Interactive CLI prompts that require user input
3. Simple utility scripts that aren't part of the main system
4. Test output that's specifically meant for console display

```python
# OK to keep print here
if __name__ == "__main__":
    print("Running consciousness demo...")
    result = demo()
    print(f"Result: {result}")
```

## Special Cases

### Quantum/Import Warnings

```python
# Before
except ImportError:
    print("Quantum module not available, using simulation")

# After
except ImportError:
    logger.warning("Quantum module not available, using simulation mode")
```

### Progress Indicators

For long-running operations, consider using DEBUG level for detailed progress:

```python
# Before
print(f"Processing {i}/{total}...")

# After
logger.debug(f"Processing {i}/{total}...")
```

## Module-Specific Setup

### For Core Modules

```python
import logging
from core.logging_config import get_logger

logger = get_logger(__name__)

class MyConsciousnessModule:
    def __init__(self):
        logger.info("Initializing consciousness module")
        # ... rest of init
```

### For Entry Points

Main entry point files should set up logging configuration:

```python
import logging
from core.logging_config import setup_logging

# Configure logging for the entire application
setup_logging(
    log_level="INFO",
    log_file="consciousness_interface",
    console_output=True,
    use_color=True,
    use_emoji=True
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Universal Consciousness Interface")
    # ... rest of main
```

## Migration Checklist

For each file you migrate:

- [ ] Add logging import at the top
- [ ] Create module logger with `get_logger(__name__)`
- [ ] Replace `print()` with appropriate `logger.level()` calls
- [ ] Choose correct log level for each message
- [ ] Keep print statements in `if __name__ == "__main__"` blocks
- [ ] Test that logging works correctly
- [ ] Update any exception handling to use logger.error()
- [ ] Add structured information where helpful

## Testing Your Changes

After migration, test with different log levels:

```bash
# Test with INFO level (default)
python your_module.py

# Test with DEBUG level (verbose)
UCI_LOG_LEVEL=DEBUG python your_module.py

# Test with ERROR level (quiet)
UCI_LOG_LEVEL=ERROR python your_module.py

# Test file logging
UCI_LOG_FILE=test_run python your_module.py
```

## Priority Files for Migration

Based on the repository analysis, prioritize these files:

**High Priority (Core System Logic):**
1. `core/enhanced_cross_consciousness_protocol.py` (40 prints)
2. `core/performance_optimizer.py` (27 prints)
3. `core/integrated_consciousness_system_complete.py` (26 prints)
4. `unified_consciousness_interface.py` (26 prints)
5. `enhanced_consciousness_chatbot_application.py` (21 prints)

**Medium Priority (Important Modules):**
- `core/enhanced_quantum_bio_integration.py` (24 prints)
- `core/enhanced_safety_ethics_framework.py` (23 prints)
- `core/bio_digital_hybrid_intelligence.py` (23 prints)
- `standalone_consciousness_ai.py` (15 prints)

**Low Priority (Demos and Tests):**
- Demo files in `demos/` directory
- Test files in `tests/` directory
- These can keep print() statements for user-facing output

## Example Migration

### Before (enhanced_safety_ethics_framework.py)

```python
def validate_consciousness_level(self, level):
    print(f"Validating consciousness level: {level}")

    if level > 0.9:
        print("WARNING: Consciousness level exceeds safe threshold!")
        return False

    print("Consciousness level validated successfully")
    return True
```

### After

```python
import logging
from core.logging_config import get_logger

logger = get_logger(__name__)

def validate_consciousness_level(self, level):
    logger.debug(f"Validating consciousness level: {level}")

    if level > 0.9:
        logger.warning(f"Consciousness level {level} exceeds safe threshold 0.9")
        return False

    logger.info(f"Consciousness level {level} validated successfully")
    return True
```

## Common Pitfalls

### ❌ Don't do this:
```python
logger.info(print(f"Value: {value}"))  # print returns None!
```

### ✅ Do this instead:
```python
logger.info(f"Value: {value}")
```

### ❌ Don't do this:
```python
# Logging inside tight loops
for item in millions_of_items:
    logger.debug(f"Processing {item}")  # Will flood logs!
```

### ✅ Do this instead:
```python
logger.info(f"Processing {len(millions_of_items)} items...")
for i, item in enumerate(millions_of_items):
    if i % 10000 == 0:  # Log every 10,000 items
        logger.debug(f"Processed {i}/{len(millions_of_items)} items")
```

## Questions?

If you're unsure about the appropriate log level or migration approach:
1. Check this guide for similar examples
2. Look at already-migrated files for patterns
3. Default to INFO level if uncertain
4. Tests and demos can keep print() statements

## Summary

- **Import:** `from core.logging_config import get_logger; logger = get_logger(__name__)`
- **Replace:** `print(msg)` → `logger.info(msg)`
- **Choose level:** DEBUG < INFO < WARNING < ERROR < CRITICAL
- **Keep print():** Only in `if __name__ == "__main__"` blocks
- **Test:** Verify logging works before committing

---

Generated as part of code quality improvements for the Universal Consciousness Interface project.
