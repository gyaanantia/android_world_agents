#!/bin/bash

# Test script to validate the android_world_test environment

echo "Testing Android World installation..."

# Check if we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "android_world_test" ]]; then
    echo "Please activate the test environment first:"
    echo "  conda activate android_world_test"
    exit 1
fi

echo "✓ In android_world_test environment"

# Test basic import
echo -n "Testing basic android_world import... "
if python -c "import android_world" 2>/dev/null; then
    echo "✅ SUCCESS"
else
    echo "❌ FAILED"
    exit 1
fi

# Test agents import
echo -n "Testing android_world.agents import... "
if python -c "import android_world.agents" 2>/dev/null; then
    echo "✅ SUCCESS"
else
    echo "❌ FAILED"
    exit 1
fi

# Test specific agent imports
echo -n "Testing android_world.agents.infer import... "
if python -c "import android_world.agents.infer" 2>/dev/null; then
    echo "✅ SUCCESS"
else
    echo "❌ FAILED"
fi

echo -n "Testing android_world.agents.t3a import... "
if python -c "import android_world.agents.t3a" 2>/dev/null; then
    echo "✅ SUCCESS"
else
    echo "❌ FAILED"
fi

echo -n "Testing android_world.agents.m3a_utils import... "
if python -c "import android_world.agents.m3a_utils" 2>/dev/null; then
    echo "✅ SUCCESS"
else
    echo "❌ FAILED"
fi

# Test basic functionality
echo -n "Testing basic functionality... "
if python -c "
import android_world
from android_world import registry
print('Registry available:', hasattr(registry, 'TaskRegistry'))
" 2>/dev/null; then
    echo "✅ SUCCESS"
else
    echo "❌ FAILED"
fi

echo ""
echo "🎉 All tests completed!"
echo ""
echo "If you want to clean up the test environment:"
echo "  conda deactivate"
echo "  conda env remove -n android_world_test"
echo "  rm -rf android_world_test_repo"
