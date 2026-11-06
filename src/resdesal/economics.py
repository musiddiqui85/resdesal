"""Cost per unit of surviving capacity and a lifecycle check on T* (Section 2.5 of the paper).

The capital-only break-even T* = c_b / c_s holds for any measure once c_b is
expressed per m³/d of output that survives the event: a premium paid on K
m³/d that delivers a × f × K during the event costs premium / (a f) per
surviving m³/d. The lifecycle version adds O&M and energy as present values
over the economic life (E17, E18).
"""

from __future__ import annotations

from dataclasses import dataclass

from .params import COSTS, Costs

DISCOUNT_RATE = 0.07  # E17, A
LIFE_YEARS = 25  # E18, S

GAS_PRICE = 2.15  # E9, S. USD/MMBtu
ELECTRICITY_PRICE = 0.053  # E10, S. USD/kWh
RO_ENERGY = 3.5  # P9, S/A. kWh/m³
MED_ELECTRICITY = 2.0  # M4, S. kWh/m³
MED_STEAM = 137.0 * 24.0 / 29_630.0  # M6 / M1, D. t steam per m³ = 0.111
STEAM_HEAT = 2.36  # A. GJ/t, 10 bar saturated steam from ~100 °C condensate
MSF_ELECTRICITY = 4.0  # F4, S. kWh/m³, range 3.0-5.0 (World Bank 2019: 3.4-4.5)
MSF_STEAM = 1.0 / 9.0  # F2/F8, D. t of brine-heater steam per m³ at GOR 9 = 0.111
MSF_STEAM_HEAT = 2.30  # F8, A. GJ/t, about 2.5 bar saturated steam from ~100 °C condensate
BOILER_EFFICIENCY = 0.90  # A
GJ_PER_MMBTU = 1.055
CCGT_STEAM_PENALTY = 12.0  # A. kWh_e lost per m³ of MED product (range 8-16)
CCGT_STEAM_PENALTY_MSF = 14.0  # F9, D/A. kWh_e lost per m³ of MSF product (range 8-19; Zeitoun et al. 2023 HRSG case 18.2-19.0)
STORAGE_OM = 0.01  # A. Fraction of storage capex per year
LAYUP = 0.003  # E7, D. Fraction of capex per year for a laid-up unit (range 0.001-0.01)
BOILER_USD_PER_KW = 120.0  # E4a, S. Gas boiler capex per kW of fuel input
FIXED_OM = 0.02  # E8, S. Fraction of capex per year
DAF_UF_OPEX = 0.02  # A. USD/m³ extra chemicals and energy


def annuity_factor(rate: float = DISCOUNT_RATE, years: int = LIFE_YEARS) -> float:
    """Present value of 1 per year."""
    return (1.0 - (1.0 + rate) ** -years) / rate


# Per thermal technology: electricity (kWh/m³), steam (t/m³), steam heat (GJ/t), CCGT penalty (kWh_e/m³).
THERMAL = {
    "med": (MED_ELECTRICITY, MED_STEAM, STEAM_HEAT, CCGT_STEAM_PENALTY),
    "msf": (MSF_ELECTRICITY, MSF_STEAM, MSF_STEAM_HEAT, CCGT_STEAM_PENALTY_MSF),
}


def steam_heat(tech: str = "med") -> float:
    """Steam heat per m³ of product, GJ."""
    _, steam, heat, _ = THERMAL[tech]
    return steam * heat


def boiler_capex(tech: str = "med") -> float:
    """Auxiliary boiler capex, USD per m³/d of thermal capacity (E4a, D)."""
    fuel_kw = steam_heat(tech) / BOILER_EFFICIENCY * 1e6 / 86_400.0
    return BOILER_USD_PER_KW * fuel_kw


def boiler_fuel_cost(gas_price: float = GAS_PRICE, tech: str = "med") -> float:
    """Gas cost of thermal steam from an auxiliary boiler, USD per m³ of product."""
    mmbtu = steam_heat(tech) / BOILER_EFFICIENCY / GJ_PER_MMBTU
    return mmbtu * gas_price


def med_energy_premium(source: str, gas_price: float = GAS_PRICE, power_price: float = ELECTRICITY_PRICE,
                       tech: str = "med") -> float:
    """Extra energy cost of MED (or MSF, ``tech="msf"``) over SWRO, USD per m³ of product, at base load."""
    electricity, _, _, penalty = THERMAL[tech]
    electric = (electricity - RO_ENERGY) * power_price
    if source == "boiler":
        return electric + boiler_fuel_cost(gas_price, tech)
    if source == "ccgt":
        return electric + penalty * power_price
    raise ValueError(source)


@dataclass(frozen=True)
class Measure:
    """A way of buying surviving capacity, per m³/d of the capacity installed."""

    label: str
    capex: float  # USD per m³/d installed (premium over SWRO if it replaces SWRO)
    survival: float  # a × f: surviving output per m³/d installed
    om_fraction: float = 0.0  # fraction of capex per year
    energy: float = 0.0  # USD per m³ of product, extra over SWRO
    utilisation: float = 0.0  # fraction of the year the extra energy is paid
    basis: str = "firm"  # "expected" when survival is a probability-weighted output (second site)

    def capex_per_surviving(self) -> float:
        return self.capex / self.survival

    def lifecycle_per_surviving(self, rate: float = DISCOUNT_RATE, years: int = LIFE_YEARS) -> float:
        af = annuity_factor(rate, years)
        annual = self.om_fraction * self.capex + self.energy * 365.0 * self.utilisation
        return (self.capex + annual * af) / self.survival


def storage_lifecycle(costs: Costs = COSTS, rate: float = DISCOUNT_RATE, years: int = LIFE_YEARS) -> float:
    """Lifecycle cost of storage, USD per m³."""
    return costs.storage_capex * (1.0 + STORAGE_OM * annuity_factor(rate, years))


def measures(costs: Costs = COSTS, bloom_survival: dict[str, float] | None = None) -> list[Measure]:
    """Surviving-capacity measures for GCRP-600 under a severe bloom (base values)."""
    f = {"daf_uf": 0.9, "site_b": 0.3, "med_process": 1.0, "msf_process": 1.0, "subsurface": 1.0,
         **(bloom_survival or {})}
    c_ro, c_med, c_msf = costs.swro_capex, costs.med_capex, costs.msf_capex
    boiler, boiler_msf = boiler_capex(), boiler_capex("msf")
    return [
        Measure("A1 DAF+UF retrofit", 0.08 * c_ro, 0.95 * f["daf_uf"], FIXED_OM, DAF_UF_OPEX, 0.95),
        Measure("A2 Subsurface intake", 0.13 * c_ro, 0.95 * f["subsurface"], FIXED_OM),
        # Site B is unaffected with probability 0.3: 0.285 is expected output, not firm capacity.
        Measure("A3 Second RO site", 0.15 * c_ro, 0.95 * f["site_b"], FIXED_OM, basis="expected"),
        Measure("A4 Hybrid MED, CCGT steam", c_med - c_ro, 0.92 * f["med_process"], FIXED_OM,
                med_energy_premium("ccgt"), 0.92),
        Measure("A5 Hybrid MED, own boiler", c_med - c_ro + boiler, 0.92 * f["med_process"], FIXED_OM,
                med_energy_premium("boiler"), 0.92),
        # A5 at half load in normal times is not listed: the plant would then deliver less than normal demand.
        Measure("Standby MED + boiler (added)", c_med + boiler, 0.92 * f["med_process"], LAYUP),
        Measure("A6 Hybrid MSF, CCGT steam", c_msf - c_ro, 0.92 * f["msf_process"], FIXED_OM,
                med_energy_premium("ccgt", tech="msf"), 0.92),
        Measure("A7 Hybrid MSF, own boiler", c_msf - c_ro + boiler_msf, 0.92 * f["msf_process"], FIXED_OM,
                med_energy_premium("boiler", tech="msf"), 0.92),
    ]
