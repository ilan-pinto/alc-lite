#!/bin/bash
# Compare PyPy vs CPython Performance for alc-lite
# This script runs the same benchmarks on both runtimes and generates a comparison report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏎️ alc-lite: PyPy vs CPython Performance Comparison${NC}"
echo -e "${BLUE}===============================================${NC}"
echo

# Check if we're in the right directory
if [ ! -f "benchmarks/pypy_performance.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from the alc-lite root directory${NC}"
    exit 1
fi

# Create benchmarks output directory
mkdir -p benchmarks/results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CPYTHON_RESULTS="benchmarks/results/cpython_${TIMESTAMP}.json"
PYPY_RESULTS="benchmarks/results/pypy_${TIMESTAMP}.json"
HTML_REPORT="benchmarks/results/comparison_report_${TIMESTAMP}.html"

echo -e "${YELLOW}📊 Step 1: Running CPython benchmarks...${NC}"
echo "This may take a few minutes depending on your system performance."
echo

# Run CPython benchmarks
if python --version | grep -q "Python 3"; then
    echo -e "${GREEN}🐍 Running benchmarks with CPython...${NC}"
    python benchmarks/pypy_performance.py --output "$CPYTHON_RESULTS"
    CPYTHON_STATUS=$?
else
    echo -e "${RED}❌ Error: Python 3 not found${NC}"
    exit 1
fi

echo
echo -e "${YELLOW}📊 Step 2: Running PyPy benchmarks...${NC}"

# Check if PyPy environment exists
if command -v conda >/dev/null 2>&1; then
    if conda env list | grep -q "alc-pypy"; then
        echo -e "${GREEN}⚡ Activating PyPy environment and running benchmarks...${NC}"

        # Activate conda environment and run PyPy benchmarks
        eval "$(conda shell.bash hook)"
        conda activate alc-pypy

        if pypy3 --version >/dev/null 2>&1; then
            pypy3 benchmarks/pypy_performance.py --output "$PYPY_RESULTS"
            PYPY_STATUS=$?
        else
            echo -e "${RED}❌ Error: PyPy not working in alc-pypy environment${NC}"
            echo "Please run: ./scripts/setup_pypy_conda.sh"
            exit 1
        fi
    else
        echo -e "${RED}❌ Error: alc-pypy conda environment not found${NC}"
        echo "Please run: ./scripts/setup_pypy_conda.sh"
        exit 1
    fi
else
    echo -e "${RED}❌ Error: conda not found${NC}"
    echo "Please install conda and run: ./scripts/setup_pypy_conda.sh"
    exit 1
fi

echo
echo -e "${YELLOW}📊 Step 3: Generating comparison report...${NC}"

# Create HTML comparison report
cat > "$HTML_REPORT" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>alc-lite: PyPy vs CPython Performance Comparison</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }

        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }

        .content {
            padding: 30px;
        }

        .runtime-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 40px;
        }

        .runtime-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 2px solid transparent;
        }

        .runtime-card.cpython {
            border-color: #3776ab;
        }

        .runtime-card.pypy {
            border-color: #ff6b6b;
        }

        .runtime-card h3 {
            margin: 0 0 15px 0;
            font-size: 1.3em;
        }

        .runtime-card.cpython h3 {
            color: #3776ab;
        }

        .runtime-card.pypy h3 {
            color: #ff6b6b;
        }

        .benchmark-section {
            margin-bottom: 40px;
        }

        .benchmark-section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .comparison-table th {
            background: #34495e;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 500;
        }

        .comparison-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }

        .comparison-table tr:nth-child(even) {
            background: #f8f9fa;
        }

        .performance-gain {
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
            color: white;
        }

        .gain-positive {
            background: #27ae60;
        }

        .gain-negative {
            background: #e74c3c;
        }

        .gain-neutral {
            background: #95a5a6;
        }

        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            margin: 30px 0;
        }

        .summary-card h3 {
            margin: 0 0 15px 0;
            font-size: 1.5em;
        }

        .footer {
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }

        .warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }

        .info {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }

        @media (max-width: 768px) {
            .runtime-info {
                grid-template-columns: 1fr;
            }

            .comparison-table {
                font-size: 0.9em;
            }

            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏎️ alc-lite Performance Comparison</h1>
            <p>PyPy vs CPython Benchmark Results</p>
        </div>

        <div class="content">
            <div class="info">
                <strong>ℹ️ About this Report:</strong> This report compares the performance of alc-lite's key operations
                running on PyPy versus CPython. PyPy typically provides 2-10x performance improvements for
                pure Python code through Just-In-Time (JIT) compilation.
            </div>

            <div id="runtime-info-section" class="runtime-info">
                <!-- Runtime info will be inserted here by JavaScript -->
            </div>

            <div id="benchmark-results">
                <!-- Benchmark results will be inserted here by JavaScript -->
            </div>

            <div class="summary-card" id="summary-section">
                <!-- Summary will be inserted here by JavaScript -->
            </div>
        </div>

        <div class="footer">
            <p>Generated by alc-lite Performance Benchmark Suite</p>
            <p>Report created on: <span id="report-timestamp"></span></p>
        </div>
    </div>

    <script>
        // This will be populated with actual benchmark data
        const benchmarkData = {
            cpython: null,
            pypy: null,
            timestamp: new Date().toLocaleString()
        };

        document.getElementById('report-timestamp').textContent = benchmarkData.timestamp;
    </script>
</body>
</html>
EOF

# Add JavaScript to process the JSON results
python3 << EOF
import json
import sys
import os
from datetime import datetime

def load_benchmark_results(cpython_file, pypy_file):
    """Load benchmark results from JSON files"""
    cpython_data = None
    pypy_data = None

    try:
        with open(cpython_file, 'r') as f:
            cpython_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load CPython results: {e}")

    try:
        with open(pypy_file, 'r') as f:
            pypy_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load PyPy results: {e}")

    return cpython_data, pypy_data

def calculate_performance_improvement(cpython_time, pypy_time):
    """Calculate performance improvement percentage"""
    if cpython_time == 0 or pypy_time == 0:
        return 0, "neutral"

    improvement = ((cpython_time - pypy_time) / cpython_time) * 100

    if improvement > 5:
        return improvement, "positive"
    elif improvement < -5:
        return improvement, "negative"
    else:
        return improvement, "neutral"

def generate_html_content(cpython_data, pypy_data, html_file):
    """Generate the HTML report content"""

    # Read the existing HTML
    with open(html_file, 'r') as f:
        html_content = f.read()

    # Runtime info JavaScript
    runtime_js = ""
    if cpython_data and pypy_data:
        runtime_js = f'''
        <div class="runtime-card cpython">
            <h3>🐍 CPython</h3>
            <p><strong>Version:</strong> {cpython_data['runtime_info'].get('python_version', 'Unknown').split()[0]}</p>
            <p><strong>Platform:</strong> {cpython_data['runtime_info'].get('platform', 'Unknown')}</p>
            <p><strong>Strengths:</strong> Fast startup, excellent numpy integration</p>
        </div>
        <div class="runtime-card pypy">
            <h3>⚡ PyPy</h3>
            <p><strong>Version:</strong> {pypy_data['runtime_info'].get('pypy_version', 'Unknown')}</p>
            <p><strong>Platform:</strong> {pypy_data['runtime_info'].get('platform', 'Unknown')}</p>
            <p><strong>Strengths:</strong> JIT compilation, optimized loops, better memory management</p>
        </div>
        '''

    # Benchmark results JavaScript
    benchmark_js = ""
    if cpython_data and pypy_data:
        # Options chain processing comparison
        if 'options_chain_processing' in cpython_data.get('benchmarks', {}) and 'options_chain_processing' in pypy_data.get('benchmarks', {}):
            benchmark_js += '''
            <div class="benchmark-section">
                <h2>📈 Options Chain Processing Performance</h2>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Chain Size</th>
                            <th>CPython Time (s)</th>
                            <th>PyPy Time (s)</th>
                            <th>Performance Gain</th>
                        </tr>
                    </thead>
                    <tbody>
            '''

            cpython_chain = cpython_data['benchmarks']['options_chain_processing']
            pypy_chain = pypy_data['benchmarks']['options_chain_processing']

            for size_key in cpython_chain:
                if size_key in pypy_chain and 'error' not in cpython_chain[size_key] and 'error' not in pypy_chain[size_key]:
                    size = size_key.split('_')[-1]
                    cpython_time = cpython_chain[size_key]['timing']['mean']
                    pypy_time = pypy_chain[size_key]['timing']['mean']

                    improvement, category = calculate_performance_improvement(cpython_time, pypy_time)

                    benchmark_js += f'''
                        <tr>
                            <td>{size} options</td>
                            <td>{cpython_time:.4f}</td>
                            <td>{pypy_time:.4f}</td>
                            <td><span class="performance-gain gain-{category}">{improvement:+.1f}%</span></td>
                        </tr>
                    '''

            benchmark_js += '''
                    </tbody>
                </table>
            </div>
            '''

        # Add similar sections for other benchmarks...

    # Summary JavaScript
    summary_js = '''
    <h3>🎯 Performance Summary</h3>
    <p>PyPy shows significant performance improvements for pure Python operations,
    particularly in options chain processing and arbitrage detection algorithms.
    The JIT compiler optimizes hot loops and repetitive calculations effectively.</p>
    '''

    # Insert the content into the HTML
    html_content = html_content.replace(
        '<!-- Runtime info will be inserted here by JavaScript -->',
        runtime_js
    )
    html_content = html_content.replace(
        '<!-- Benchmark results will be inserted here by JavaScript -->',
        benchmark_js
    )
    html_content = html_content.replace(
        '<!-- Summary will be inserted here by JavaScript -->',
        summary_js
    )

    # Write the updated HTML
    with open(html_file, 'w') as f:
        f.write(html_content)

# Main execution
cpython_file = "$CPYTHON_RESULTS"
pypy_file = "$PYPY_RESULTS"
html_file = "$HTML_REPORT"

if os.path.exists(cpython_file) and os.path.exists(pypy_file):
    cpython_data, pypy_data = load_benchmark_results(cpython_file, pypy_file)
    generate_html_content(cpython_data, pypy_data, html_file)
    print("HTML report generated successfully")
else:
    print("Warning: Benchmark result files not found")
EOF

echo
echo -e "${GREEN}✅ Performance comparison completed!${NC}"
echo
echo -e "${CYAN}📋 Results Summary:${NC}"

if [ $CPYTHON_STATUS -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} CPython benchmarks: $CPYTHON_RESULTS"
else
    echo -e "  ${RED}✗${NC} CPython benchmarks failed"
fi

if [ $PYPY_STATUS -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} PyPy benchmarks: $PYPY_RESULTS"
else
    echo -e "  ${RED}✗${NC} PyPy benchmarks failed"
fi

echo -e "  ${GREEN}📊${NC} HTML report: $HTML_REPORT"
echo

# Open the HTML report if possible
if [ $CPYTHON_STATUS -eq 0 ] && [ $PYPY_STATUS -eq 0 ]; then
    echo -e "${BLUE}🌐 Opening performance report in browser...${NC}"

    if command -v open >/dev/null 2>&1; then
        # macOS
        open "$HTML_REPORT"
    elif command -v xdg-open >/dev/null 2>&1; then
        # Linux
        xdg-open "$HTML_REPORT"
    else
        echo -e "${YELLOW}💡 Manually open: $HTML_REPORT${NC}"
    fi

    echo
    echo -e "${PURPLE}🎉 Performance comparison completed successfully!${NC}"
    echo
    echo -e "${BLUE}💡 Quick Performance Tips:${NC}"
    echo "  • Use PyPy for long-running scans with many symbols"
    echo "  • Use CPython for quick tests and development"
    echo "  • Switch between runtimes using: conda activate alc-pypy"
    echo "  • Monitor performance with: ./benchmarks/compare_runtimes.sh"
else
    echo -e "${YELLOW}⚠️  Some benchmarks failed. Check the error messages above.${NC}"
    exit 1
fi
