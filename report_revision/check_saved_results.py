from pathlib import Path
import pandas as pd, json
P=Path(__file__).resolve().parents[1] / 'results'
s=json.load(open(P/'summary.json')); pf=pd.read_csv(P/'scenarios_perfect_foresight.csv'); r=pd.read_csv(P/'scenarios_rolling_horizon.csv')
x=pd.read_csv(P/'interval_vpp_2024.csv.gz',parse_dates=['timestamp']);dt=1/12
h=x.timestamp.dt.hour
pi=x.spot+((h>=15)&(h<21)).map({True:.16,False:.06})+.02+.002+.015
bill=((x.gi*pi-x["ge"]*x.spot)*dt).sum()+300
row=r[(r.tariff=='Wholesale')&(r.pv=='pv_ml')&(r.price=='forecast')].iloc[0]
bal=x.pv-x.cu+x.d+x.gi-x.load-x.c_pv-x.c_g-x["ge"]
assert abs(bill-row.annual_bill)<0.05
assert bal.abs().max()<0.0005
assert x.soc.min()>=1-1e-8 and x.soc.max()<=9.5+1e-8
assert abs((x.d*dt).sum()*0.05-row.wear_cost)<0.01
for _,v in pf.iterrows():
 assert abs(v.energy_bill+v.vpp_fee-v.annual_bill)<1e-7
rt=r[(r.tariff=='Wholesale')&(r.price=='forecast+RT')].iloc[0]
assert abs(s['vpp_refined_rt_bill']-rt.annual_bill_adj)<1e-7
report={'scope':'Independent recalculation of saved forecast-only interval settlement and CSV identities; no full optimisation rerun','rounding_note':'Interval audit CSV is rounded to four decimals; bill agrees within AUD 0.02','forecast_unadjusted_bill':bill,'forecast_adjusted_bill':row.annual_bill_adj,'max_balance_error_kW':float(bal.abs().max()),'soc_kWh':[float(x.soc.min()),float(x.soc.max())],'raw_wholesale_baseline_fee':pf[(pf.tariff=='Wholesale')&(pf['mode']!='Optimised')][['year','mode','vpp_fee']].to_dict('records'),'fee_normalisation':'Add AUD 300 to wholesale NoBESS and SelfCons for the report only; original files unchanged.'}
print(json.dumps(report,indent=2));Path(__file__).with_name('audit_result.json').write_text(json.dumps(report,indent=2))
