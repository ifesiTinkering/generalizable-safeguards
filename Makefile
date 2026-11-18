.PHONY: verify results test report package
verify:
	safeguards verify --full
results:
	safeguards reproduce
test:
	python -m pytest -q
report:
	python scripts/build_run_report.py
package:
	python scripts/package_review.py
