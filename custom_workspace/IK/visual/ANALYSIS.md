Removing the outlier was a massive success. The graphs are much cleaner, and now, instead of just seeing "bad data," we can actually see the **strategies** the subjects are using.

Here is the breakdown of the new insights revealed by your clean dataset.

### 1. The "Trunk Strategy" Trade-off (Scatter Plot)

**Graph:** `advanced_3_efficiency_scatter.png`

Now that the axis isn't squashed by the outlier, a fascinating pattern has emerged regarding how subjects use their torso:

* **The "Healthy" Cluster (Dark Purple, Top-Left):** These trials have **Low Trunk Compensation** (dark color), **High Efficiency** (High Y-axis), and **Straight Reaches** (Low X-axis). This is the ideal movement: using the arm joints effectively to reach in a straight line.
* **The "Cheater" Cluster (Yellow/Green, Middle-Left):** Look at the yellow dots. They have **High Trunk Compensation** (~160mm movement), yet they still achieve **Straight Reaches** (Low X-axis).
* **Insight:** These subjects are "cheating." They cannot extend their elbow efficiently (hence the lower Y-axis position), so they lean forward with their trunk to make the hand path look straight. This is a classic compensatory strategy in stroke rehabilitation.



### 2. The "Place" Phase is the Bottleneck

**Graphs:** `analysis_3_boxplots.png` and `advanced_2_feature_distributions.png`

With the outlier gone, the variability in the **Place** phase stands out even more sharply as the main area of difficulty.

* **High Error:** The "Place Error (mm)" boxplot shows a huge spread. While the median error is low (~30mm), the whiskers extend past 120mm. This means many subjects are "missing" the target or dropping the object inconsistently.
* **Variable Velocity:** In the standard boxplots, the "Place" phase has the widest interquartile range for Peak Velocity. Some subjects place it slowly and carefully; others likely drop it using gravity (fast).

### 3. Segmentation is Systemic (Cyclograms)

**Graph:** `advanced_1_coordination_cyclograms.jpg`

Even without the outlier, the cyclograms remain "messy," which is actually a **good finding**. It confirms that the lack of smooth coordination is a feature of your patients, not a data error.

* **The "Hook" Pattern:** Notice how many lines go straight up (Elbow moves, Shoulder locked) and then turn right (Shoulder moves, Elbow locked). This "Etch-a-Sketch" style of movement (decomposition) is a strong marker of motor impairment. If these were healthy controls, you would see smooth, diagonal ellipses.

### 4. New "Soft" Outliers Identified (Heatmap)

**Graph:** `analysis_5_heatmap.png`

The heatmap dynamic range is fixed, revealing nuanced differences between specific files:

* **File `S1_12_3` (Row 11):** Dark Red in `Reach_PeakVel` and `Reach_PathLen`. This subject moved very fast and took a long path, possibly overshooting the target.
* **File `S5_12_3` (Last Row):** Dark Red in `Lift_PeakVel` and `Place_PeakVel`. This subject seems to be performing the task much more aggressively/quickly than the others.

### Recommendation

Your data is now "analysis-ready." You have successfully separated **noise** (the removed outlier) from **pathology** (the trunk compensation and poor coordination).

**Next Step for your ML/Paper:**
Group your files into two classes based on the Scatter Plot results:

1. **Arm Strategy:** Low Trunk Comp (<60mm) + High Efficiency.
2. **Trunk Strategy:** High Trunk Comp (>100mm).

This binary classification will likely yield very strong results if you try to train a classifier on this dataset.