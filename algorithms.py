from zellij.core import Threshold, IThreshold, Calls, BooleanStop

from zellij.strategies.tools import Hypersphere, Section, Direct

from zellij.strategies.tools import (
    Move_up,
    Potentially_Optimal_Rectangle,
    Locally_biased_POR,
    Adaptive_POR,
    Soo_tree_search,
    Best_first_search,
    Cyclic_best_first_search,
    Beam_search,
)
from zellij.strategies.tools import (
    Distance_to_the_best,
    Min,
    Distance_to_the_best_centered,
    Nothing,
)
from zellij.strategies.tools import Sigma2, SigmaInf
from zellij.strategies import PHS, ILS, DirectSampling, DBA, DBA_Direct, CenterSOO

from zellij.utils import Basic


def FDA_souquet(values, loss, inflation=1.75, level=5, convert=True):
    if convert:
        sp = Hypersphere(values, loss, converter=Basic())
    else:
        sp = Hypersphere(values, loss)

    phs = PHS(sp, inflation=inflation)
    ils = ILS(sp, inflation=inflation)

    stop1 = BooleanStop(phs, "computed")
    stop2 = IThreshold(ils, "step", 1e-20)

    fda = DBA(
        sp,
        Move_up(sp, level),
        exploration=(phs, stop1),
        exploitation=(ils, stop2),
        scoring=Distance_to_the_best(),
    )

    return fda


def FDA_souquet_D(
    values, loss, inflation=1.75, level=10, save_close=False, convert=True
):
    if convert:
        sp = Hypersphere(values, loss, converter=Basic())
    else:
        sp = Hypersphere(values, loss)

    phs = PHS(sp, inflation=inflation)
    ils = ILS(sp, inflation=inflation)

    stop1 = BooleanStop(phs, "computed")
    stop2 = IThreshold(ils, "step", 1e-20)

    fda = DBA(
        sp,
        Move_up(sp, level, save_close=save_close),
        exploration=(phs, stop1),
        exploitation=(ils, stop2),
        scoring=Distance_to_the_best(),
    )

    return fda


def FDA_souquet_centered(
    values, loss, inflation=1.75, level=5, save_close=False, convert=True
):
    if convert:
        sp = Hypersphere(values, loss, converter=Basic())
    else:
        sp = Hypersphere(values, loss)

    phs = PHS(sp, inflation=inflation)
    ils = ILS(sp, inflation=inflation)

    stop1 = BooleanStop(phs, "computed")
    stop2 = IThreshold(ils, "step", 1e-20)

    fda = DBA(
        sp,
        Move_up(sp, level, save_close=save_close),
        exploration=(phs, stop1),
        exploitation=(ils, stop2),
        scoring=Distance_to_the_best_centered(),
    )

    return fda


def Direct_c(
    values, loss, level=600, error=1e-4, maxdiv=10000, save_close=False, convert=True
):
    if convert:
        sp = Direct(values, loss, measure=Sigma2(), converter=Basic())
    else:
        sp = Direct(values, loss, measure=Sigma2())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")

    dba = DBA_Direct(
        sp,
        Potentially_Optimal_Rectangle(
            sp, max_depth=level, error=error, maxdiv=maxdiv, save_close=save_close
        ),
        (explor, stop1),
        scoring=Nothing(),
    )

    return dba


def Direct_l(
    values, loss, level=600, error=1e-4, maxdiv=10000, save_close=False, convert=True
):
    if convert:
        sp = Direct(values, loss, measure=SigmaInf(), converter=Basic())
    else:
        sp = Direct(values, loss, measure=SigmaInf())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")

    dba = DBA_Direct(
        sp,
        Locally_biased_POR(
            sp, max_depth=level, error=error, maxdiv=maxdiv, save_close=save_close
        ),
        (explor, stop1),
        scoring=Nothing(),
    )

    return dba


def Direct_restart(
    values, loss, level=600, error=1e-2, maxdiv=10000, save_close=False, convert=True
):
    if convert:
        sp = Direct(values, loss, measure=Sigma2(), converter=Basic())
    else:
        sp = Direct(values, loss, measure=Sigma2())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")

    dba = DBA_Direct(
        sp,
        Adaptive_POR(
            sp, max_depth=level, error=error, maxdiv=maxdiv, save_close=save_close
        ),
        (explor, stop1),
        scoring=Nothing(),
    )

    return dba


def Soo_c(values, loss, level=600, section=3, save_close=False, convert=True):
    if convert:
        sp = Section(values, loss, section=section, converter=Basic())
    else:
        sp = Section(values, loss, section=section)

    explor = CenterSOO(sp)
    stop1 = BooleanStop(explor, "computed")
    dba = DBA(
        sp,
        Soo_tree_search(sp, level, save_close=save_close),
        (explor, stop1),
        scoring=Min(),
    )

    return dba


def FDA_souquet_Bfs(
    values, loss, inflation=1.75, level=5, save_close=False, convert=True
):
    if convert:
        sp = Hypersphere(values, loss, converter=Basic())
    else:
        sp = Hypersphere(values, loss)

    phs = PHS(sp, inflation=inflation)
    ils = ILS(sp, inflation=inflation)

    stop1 = BooleanStop(phs, "computed")
    stop2 = IThreshold(ils, "step", 1e-20)

    dba = DBA(
        sp,
        Best_first_search(sp, level, save_close=save_close),
        exploration=(phs, stop1),
        exploitation=(ils, stop2),
        scoring=Distance_to_the_best(),
    )

    return dba


def FDA_souquet_DBfs(
    values, loss, inflation=1.75, level=10, save_close=False, convert=True
):
    if convert:
        sp = Hypersphere(values, loss, converter=Basic())
    else:
        sp = Hypersphere(values, loss)

    phs = PHS(sp, inflation=inflation)
    ils = ILS(sp, inflation=inflation)

    stop1 = BooleanStop(phs, "computed")
    stop2 = IThreshold(ils, "step", 1e-20)

    dba = DBA(
        sp,
        Best_first_search(sp, level, save_close=save_close),
        exploration=(phs, stop1),
        exploitation=(ils, stop2),
        scoring=Distance_to_the_best(),
    )

    return dba


def Direct_Bfs(values, loss, level=600, save_close=False, convert=True):
    if convert:
        sp = Direct(values, loss, measure=Sigma2(), converter=Basic())
    else:
        sp = Direct(values, loss, measure=Sigma2())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")

    dba = DBA_Direct(
        sp,
        Best_first_search(sp, level, save_close=save_close),
        (explor, stop1),
        scoring=Nothing(),
    )

    return dba


def FDA_souquet_Bs(
    values, loss, inflation=1.75, level=5, save_close=False, convert=True
):
    if convert:
        sp = Hypersphere(values, loss, converter=Basic())
    else:
        sp = Hypersphere(values, loss)

    phs = PHS(sp, inflation=inflation)
    ils = ILS(sp, inflation=inflation)

    stop1 = BooleanStop(phs, "computed")
    stop2 = IThreshold(ils, "step", 1e-20)

    dba = DBA(
        sp,
        Beam_search(sp, level, beam_length=6000, save_close=save_close),
        exploration=(phs, stop1),
        exploitation=(ils, stop2),
        scoring=Distance_to_the_best(),
    )

    return dba


def FDA_souquet_DBs(
    values, loss, inflation=1.75, level=10, save_close=False, convert=True
):
    if convert:
        sp = Hypersphere(values, loss, converter=Basic())
    else:
        sp = Hypersphere(values, loss)

    phs = PHS(sp, inflation=inflation)
    ils = ILS(sp, inflation=inflation)

    stop1 = BooleanStop(phs, "computed")
    stop2 = IThreshold(ils, "step", 1e-20)

    dba = DBA(
        sp,
        Beam_search(sp, level, beam_length=6000, save_close=save_close),
        exploration=(phs, stop1),
        exploitation=(ils, stop2),
        scoring=Distance_to_the_best(),
    )

    return dba


def Direct_Bs(values, loss, level=600, save_close=False, convert=True):
    if convert:
        sp = Direct(values, loss, measure=Sigma2(), converter=Basic())
    else:
        sp = Direct(values, loss, measure=Sigma2())

    explor = DirectSampling(sp)
    stop1 = BooleanStop(explor, "computed")

    dba = DBA_Direct(
        sp,
        Beam_search(sp, level, beam_length=6000, save_close=save_close),
        (explor, stop1),
        scoring=Nothing(),
    )

    return dba


def Soo_Bs(values, loss, level=600, section=3, save_close=False, convert=True):
    if convert:
        sp = Section(values, loss, section=section, converter=Basic())
    else:
        sp = Section(values, loss, section=section)

    explor = CenterSOO(sp)
    stop1 = BooleanStop(explor, "computed")
    dba = DBA(
        sp,
        Beam_search(sp, level, beam_length=6000, save_close=save_close),
        (explor, stop1),
        scoring=Min(),
    )

    return dba
