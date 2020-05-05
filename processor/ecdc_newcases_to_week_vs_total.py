# 将ecdc的new_case.csv格式化成"每SUM_WINDOW天新增 vs Total"

import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import interp1d, splrep, splev
import math
import numpy as np
import pandas as pd
import xlsxwriter
import seaborn as sns

#%matplotlib inline

ECDC_FILE = '../covid-19-data/public/data/ecdc/new_cases.csv'

X_STEP = 1000
SUM_WINDOW = 7
#COLS = {'Canada','China', 'France','Germany', 'India', 'Japan', 'Russia', 'Spain', 'United Kingdom'}

COLS = {'France', 'Germany', 'Italy', 'Russia', 'United Kingdom'}
#COLS = {'Canada'}

'''
COLS = {'World', 'Australia', 'Canada', 'China', 'France', \
                                'Germany', 'India', 'Indonesia', 'Iran', 'Italy', 'Japan', \
                                'Malaysia', 'New Zealand', 'Russia', 'Singapore', 'South Korea', \
                                'Spain', 'United Kingdom', 'United States' \
                                }
'''

'''
newcases = pd.read_csv(ECDC_FILE, delimiter=',', header=0, \
                       usecols = COLS)
'''

newcases = pd.read_csv(ECDC_FILE, delimiter=',', header=0, \
                       usecols = COLS)


#npcases = np.array(newcases)
#print(npcases)

#pd.set_option('display.max_columns', None)
np.printoptions(precision=0, suppress=False)
newcases.fillna(0)

#print(newcases[['World']].head(20))
#cs = newcases[['World']].cumsum()
#print(cs.head(20))

print(newcases.tail(10))
total_till_date = newcases.cumsum()
print(total_till_date.tail(10))

print(newcases.tail(20))
prev_days_total = newcases.rolling(min_periods=1, window=SUM_WINDOW).sum()
print(prev_days_total.tail(20))

#y_axis_max = total_till_date[['United States']].max()
# 计算x轴的最大值, 即国家的case总数
x_axis_all_countries_max = total_till_date.max()
x_axis_max_prev = x_axis_all_countries_max.max()
digits = int(math.log10(x_axis_max_prev))+1
print('x_axis_max_prev =', x_axis_max_prev)
x_axis_max = round(round(x_axis_max_prev/(10**(digits-1))+0.1,1)*(10**(digits-1)), 0)
print('x_axis_max =', x_axis_max)

# 计算y轴的最大值，即过去num天累计case数量
y_axis_all_countries_max = prev_days_total.max()
y_axis_max_prev = y_axis_all_countries_max.max()
digits = int(math.log10(y_axis_max_prev))+1
print('y_axis_max_prev =', y_axis_max_prev)
y_axis_max = round(round(y_axis_max_prev/(10**(digits-1))+0.1,1)*(10**(digits-1)), 0)
print('y_axis_max =', y_axis_max)

# Set index as incremental of 0-x_axis_max with step
col_total = np.arange(0, x_axis_max+X_STEP, X_STEP)
print(col_total)

newcases_vs_total = pd.DataFrame(columns=newcases.columns, data=None, index=col_total)
#newcases_vs_total = newcases_vs_total.rename(columns={"World": "Total Axis"})
#newcases_vs_total['Total Axis'] = col_total.tolist()
print(newcases_vs_total)
#print(newcases_vs_total[['Total Axis']])

master_idx = newcases_vs_total.index.array
print('master_idx=\n', master_idx)
for key, value in newcases_vs_total.iteritems():
    if key == "Total Axis":
        continue
    print('Processing', key)

    #for row in value:
    #    print('row=', row)

    #country's total case and previous days total
    #df_total = total_till_date[key]
    #df_prevsum = prev_days_total[key]
    #print(df_total.max())
    #print(df_total)

    max = total_till_date[key].max()
    # loop 每个刻度
    for axis_value in master_idx:
        # 如果刻度值超过当前国家最大值，则中断此次执行，进入下一个国家
        if axis_value > max:
            break;
        idx = total_till_date[key].sub(axis_value).abs().idxmin()
        newcases_vs_total.loc[axis_value, key] = prev_days_total[key][idx]
    #break

print('newcases_vs_total=\n', newcases_vs_total.head(50))

writer = pd.ExcelWriter('newcases_vs_total.xlsx', engine='xlsxwriter')
newcases_vs_total.to_excel(writer, sheet_name='weekly trend')
writer.save()

#newcases_vs_total[newcases_vs_total.columns.difference(['United States'])].plot()
#newcases_vs_total[newcases_vs_total.columns.difference(['United States', 'France', 'Italy', 'Spain'])].plot()

#spline smooth
'''
T = newcases_vs_total[newcases_vs_total.columns.difference(['United States'])]
x = T
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
xnew = np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
'''

fig = plt.figure()

fig.suptitle('weekly new cases vs total', fontsize=14)
plt.xlabel('Total', fontsize=11)
plt.ylabel('weekly new cases', fontsize=11)

#list_y = newcases_vs_total[newcases_vs_total.columns]['Canada']
countries = newcases_vs_total[COLS]
#countries = newcases_vs_total.columns.difference(['New Zealand', 'United States'])
#countries = newcases_vs_total.columns.difference(['New Zealand'])
#countries = newcases_vs_total.columns.difference( \
#    ['New Zealand', 'United States', 'Spain', 'Italy', 'United Kingdom', 'Germany', 'France'\
#     ])
for country in countries:
    print('Prepare to plot', country, '...')

    #list_y = newcases_vs_total['Canada'].values
    #list_x = newcases_vs_total.index.values
    #list_y = newcases_vs_total['Spain'].dropna().to_numpy().astype(int)
    list_y = newcases_vs_total[country].dropna().to_numpy().astype(int)
    list_x = newcases_vs_total.index.to_numpy()[:list_y.size].astype(int)

    #newcases_vs_total[country].plot()

    '''
    poly = np.polyfit(list_x,list_y,5)
    poly_y = np.poly1d(poly)(list_x)
    line, = plt.plot(list_x,poly_y, label=country)
    '''

    '''
    x_new = np.linspace(list_x.min(), list_x.max(), 100)
    f = interp1d(list_x, list_y, kind='quadratic')
    y_smooth = f(x_new)
    plt.plot(x_new, y_smooth, label=country)
    #plt.scatter(list_x, list_y)
    '''

    #'''
    f = interp1d(list_x, list_y)
    f2 = interp1d(list_x, list_y, kind='cubic')
    xnew = np.linspace(list_x.min(), list_x.max(), num=300, endpoint=True)
    #plt.plot(list_x, list_y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
    plt.plot(xnew, f(xnew), label=country)
    #plt.plot(xnew, f2(xnew), label=country)
    #'''

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(right=0.8)
plt.show()