# BAL14 replay path compatibility sidecar

This standard-library-only tool treats Windows/POSIX provenance paths as source
labels, independently of the host path parser. It requires unique exact runtime
basename coverage and validates schema, source head, raw hashes and actual sizes
against the unchanged logical-record index and packaged runtime bytes.

Legacy v2 manifests did not record external file sizes. No legacy record is
rewritten to invent them: sizes are verified from original logical records and
actual container files. Optional size fields, if supplied, must agree.

File reads use only a confined container root and a trusted basename allowlist;
source labels never select arbitrary files or execute code. Scientific modules,
arrays, numerical rules and historical evidence are unchanged. The new revision
manifest and original Linux failure remain separate from the original package.
