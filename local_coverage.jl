using LocalCoverage
lcov = generate_coverage(run_test=true)
html_coverage(lcov; open=true)