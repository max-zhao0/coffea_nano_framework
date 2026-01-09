# Luminosity Scaling

(Edition: November 5)

We scale each process to the CMS recorded luminosity $\mathcal{L}$ ([Run2](https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2#Quick_summary_table), [Run3](https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA)). We have that the total number of events expected from a certain process $p$ is given by
$$
    N_{exp}^p = \sigma_p \mathcal{L}
$$
But, from MonteCarlo we get a different number of events $N_{MC}^p = \sum_i \prod_j w^p_{ij}$, where the sum is over MC events and the product is over different kinds of weights, e.g. $j=\text{GenWeight},\text{PileUp},\text{EWCorrections},...$. Then, to obtain $N_{exp}^p$, we apply a global normalization by
$$
    w_{\mathcal{L}} = \frac{\sigma_p \mathcal{L}}{N_{MC}^p}
$$
Therefore, here in `data/LumiWeight/` we store the cross sections and luminosities needed. Since the total number of MC events can change depending on selection, that is stored under `config/`. This readme is used to keep the source of luminosities and cross sections.

## Luminosities

Given in `luminosity.json` in $\text{pb}^{-1}$ for:
- [Run2](https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2#Quick_summary_table)
- [Run3](https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis#DATA) (From Golden JSON)

## Cross-Sections

For Run3, they are given in `cross_sections.json` in $\text{pb}$. For
- [$\mathrm{t\bar{t}}$](https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO)
- Single top [$\mathrm{tW}$](https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopNNLORef#Single_top_quark_tW_channel_cros) (here stored is half total), [$t$](https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopNNLORef#Single_top_quark_t_channel_cross) and $s$ channel

**Note**: Other processes are listed, obtained from excel on `data/LumiWeight`, Jack got that.

## Branching Fractions

Sometimes we study separate channels of the same process, e.g. $pp\to\mathrm{t\bar{t}}$ is the main process, but we could split it into its different decay modes, for which
$$
    \sigma_p = \sum_i \Gamma_i \sigma_{ip}
$$
where $\sigma_{ip}$ is the cross section for that decay mode, and $\Gamma_i$ is the branching fractions (or probability) of such decay mode.

For Run2, some are given in `branching_fractions_runII.json`. Not only for different decay modes, but also for different generators to compute corrections. For example,
- $\mathrm{t\bar{t}}$
- [POWHEG Leptons](https://github.com/cms-sw/genproductions/blob/master/bin/Powheg/production/2017/13TeV/TT_hvq/TT_hdamp_NNPDF31_NNLO_dilepton.input)
- [PDG $\mathrm{W}$ leptonic BR](https://pdg.lbl.gov/2024/listings/rpp2024-list-w-boson.pdf)
- [PDG $\tau$ leptonic BR](https://pdg.lbl.gov/2024/listings/rpp2024-list-tau.pdf)