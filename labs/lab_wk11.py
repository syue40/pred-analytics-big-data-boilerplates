from matplotlib import pyplot as plt
import pylab as plb
plb.rcParams['font.size'] = 16

def mk_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups

def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

def label_group_bar(ax, data):
    groups = mk_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)

    ax.bar(xticks, y, align='center')
    ax.set_xticks(xticks)
    ax.set_xticklabels(x,)
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True)

    scale = 1. / ly
    for pos in range(ly + 1):
        add_line(ax, pos * scale, -.1)
    ypos = -.2 # Adjust this to shift the bottom labels
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes, rotation=70,
                    color='red')
            add_line(ax, pos * scale, ypos)
            pos += rpos
        add_line(ax, pos * scale, ypos)
        ypos -= .1

dataDict = {
    # Set attributes and level part worths for the attribute.
    "Car Safety": { "High":5, "Good":5, "Average":5},
    "Fuel": { "18km/l":8, "15km/l":5, "13km/l":2},
    "Accessories": { "Roadside Assist":4, "BackupCam":6, "Heated Seats":5},

}
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(1, 1, 1)
label_group_bar(ax, dataDict)
plt.title("Part Worths")
plt.show()

import matplotlib.pyplot as plt

carAttributes = ['Saftey', 'Fuel Economy', 'Accessories']
importanceLevels = [0,75,25]
plt.bar(carAttributes, importanceLevels)
plt.xticks(rotation=75)
plt.title("Car Attribute Importance")
plt.show()
