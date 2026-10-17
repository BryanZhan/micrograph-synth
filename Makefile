.PHONY: test e2e showcase

test:
	python -m pytest -q

e2e:
	python -m micrograph_synth.cli --preset grains --count 1 --size 128 --out /tmp/mg-out

showcase:
	python tools/make_showcase.py
