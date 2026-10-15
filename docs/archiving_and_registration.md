# Archiving and registration

## Zenodo (code and results, with a DOI)

1. Sign in to zenodo.org with GitHub, open *GitHub* in the account menu and switch on `M3R2T2L2/bovw_forensics`.
2. On GitHub, create a release (tag `v0.2.0`). Zenodo archives that release and mints a DOI, using `.zenodo.json`
   for the title, author, licence and description.
3. Add the DOI badge to the README and the DOI to `CITATION.cff` (`doi:` field).

Before tagging, commit the pinned environment from the Colab run (`requirements-colab.lock`, written by the last
cell of notebook 10) so the archive records the exact package versions.

## OSF (preregistrations)

The VisA, 3CAD and hybrid preregistrations were committed to git, whose dates an author can set. From now on:

1. Create an OSF project "BoVW Forensics" and link the GitHub repository (Add-ons > GitHub).
2. For each new study, write the preregistration, then create a registration with the *OSF Preregistration*
   template **before** any run that could reveal results. OSF timestamps it and freezes it.
3. Put the OSF registration link in the preregistration file and in the write-up.

The three existing preregistrations can be uploaded to the project as historical documents, with their commit
hashes, but should not be registered as if they were new: describe them as git-timestamped.
