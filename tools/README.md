# Local document tools

The `antiword` and `catdoc` packages are installed locally from Debian packages because this account cannot use passwordless `sudo`. Their extracted files are ignored by Git and are used only by `scripts/build_corpus.py` for legacy `.doc` parsing.

The package archives and extracted trees are local installation artifacts. Recreate them with:

```bash
mkdir -p tools/antiword tools/catdoc
(cd tools/antiword && apt-get download antiword && dpkg-deb -x antiword_*.deb root)
(cd tools/catdoc && apt-get download catdoc && dpkg-deb -x catdoc_*.deb root)
```
