---
layout: post
title: Earthquake Tracking Dashboard Using Tableau
image: "/posts/earthquake_header_new.png"
tags: [Tableau, Data Viz]
---

In this project, I'm going to discuss how we can use Tableau to create an earthquake tracking dashboard in order to help our client better analyse and visualise global earthquake patterns.

<br>

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
- [01. Data Overview](#data-overview)
- [02. Building The Features](#build-features)
- [03. Creating The Dashboard](#create-dashboard)

<br>

___

# Project Overview  <a name="overview-main"></a>

<br>

### Context <a name="overview-context"></a>

Our client would like to better analyse and visualise global earthquake patterns. To do this, they have provided us with a 30-day sample of data and have asked that we build an initial proof of concept. This would need to take the form of a single dashboard which includes the following features:

* A list of the top 10 largest earthquakes
* A breakdown of the percentage of earthquakes that occurred in each broad location
* For more granular locations, a view which shows how many earthquakes took place, what the average magnitude was, and what the maximum magnitude was for each respective location
* A map showing where earthquakes took place, which also shows their intensity in terms of magnitude

Another key requirement for our client is to be able to find patterns in the data. Therefore, they would also like to have use of a single date filter attached to all the above features. This will allow them to manually scroll through one day at a time, updating everything as they go, in order to track earthquakes day by day.

<br>

### Actions <a name="overview-actions"></a>

We will be using Tableau to build the dashboard and so our first task is to import the sample data provided into the tool and ensure that we understand the data we have available to work with.

Once we have a clear picture of the data, we will then look to create each of the 4 required features on a new worksheet.

Finally, we will bring each of our elements together into a single dashboard and link them all to a single date filter.

<br>

### Results <a name="overview-results"></a>

Below you can view and interact with the completed Tableau Dashboard that tracks global earthquake activity across a 30-day period:

<iframe seamless frameborder="0" src="https://public.tableau.com/shared/4M22FRNQ7?:embed=yes&:display_count=yes&:showVizHome=no" width = '1090' height = '900'></iframe>

<br>
<br>

___

# Data Overview  <a name="data-overview"></a>

As our data was provided as a csv file, it is a simple task to import this. Once we have created our new workbook, we can select *Connect to Data > Text file* before navigating to the relevant file and opening this in Tableau.

Once we have imported our data, we can assess the data itself and ensure all data types are accurate. On this occasion, we can see that all of these are correct and include strings, date and time, whole numbers, and decimal numbers which are further identified with geographic roles of latitude and longitude.

Luckily for us, our client has provided us with a clean sample of data, so no further actions are required at this stage, although we will later look to create some calculated fields.

A sample of the data (the first 10 rows) can be seen below:

| **Id** | **Datetime** | **Latitude** | **Longitude** | **Magnitude** | **Location** | **Location-Broad** |
|---|---|---|---|---|---|---|
| 10001	| 11/07/2022 15:39	| 19.19400024 | -155.491333 | 6 | Hawaii | North America |
| 10002	| 11/07/2022 15:45	| 19.17499924 | -155.5010071 | 7 | Hawaii | North America |
| 10003	| 11/07/2022 16:21	| 19.15099907 | -155.4741669 | 5 | Hawaii | North America |
| 10004	| 11/07/2022 16:22	| 19.17483333 | -155.4843333 | 11 | Hawaii | North America |
| 10005	| 11/07/2022 16:39	| 1.4277 | 124.0941 | 36 | Indonesia | Asia |
| 10006	| 11/07/2022 16:47	| 38.684 | -116.1046 | 22 | Nevada | North America |
| 10007	| 11/07/2022 16:49	| 41.8158 | -83.5123 | 8 | Michigan | North America |
| 10008	| 11/07/2022 17:16	| 26.6588 | 54.2887 | 42 | Iran | Middle East |
| 10009	| 11/07/2022 17:23	| 35.11516667 | -95.52833333 | 9 | Oklahoma | North America |
| 10010	| 11/07/2022 17:32	| 26.5869 | 54.2687 | 36 | Iran | Middle East |

<br>
<br>

___

# Building The Features  <a name="build-features"></a>

<br>

#### Top 10 Largest Earthquakes

To kick off, we will create our top 10 biggest earthquakes chart. After creating a new worksheet, we can simply drag the *Id* variable onto the Rows area of our chart, followed by the *Magnitude* variable onto the Values area, and after sorting by descending values, we already have a useful table.

To ensure our list is only showing the top 10 largest earthquakes we will create a calculated field. Our calculation is as follows:

```
INDEX() <= 10
```

The use of INDEX here is a table calculation that returns the row number and effectively means our newly calculated field acts as a Boolean True/False variable. Now that this has been created, we can drag it onto our Filters section and select "True" in the pop-up window to ensure we only see rows 10 and under in the table.

Next, we can add *Location* to our table, and we will do this by dragging the variable onto our Rows area. We then need to format our *Magnitude* column. We will first drag over *Measure Names* to our values column to give it a title, remembering to remove the second instance of the variable in our Columns section that is added automatically. We can then click into our *Measure Values* icon in the Marks section to format our values as whole numbers.

Our final action is then to add our date filter. To do so, we drag the *Datetime* variable to the Filters section. We will select *Month / Day / Year* from our filter options and select all dates. Once we have shown our filter, we will also display this as a "Single Value (slider)" so our client can easily scroll through the dates as requested.

![alt text](/img/posts/top-ten-earthquakes.jpg "Table of Top 10 Largest Earthquakes")

<br>

#### Percentage of Earthquakes by Location Broad

For our next feature, we are again going to create a table. On a new worksheet, we will drag our *Location-Broad* variable onto the Rows section. To add the percentages, however, we're going to need another calculated field. Our first approach here might be to start with a count of the *Id* variable before adjusting to percentage values, but as this is being treated as a dimension, rather than a measure, Tableau is unable to count this effectively. At this point, we could just convert *Id* to a measure. However, it may be useful to have the *Id* variable available as a dimension for other visualisations and, in fact, we have already used it as such in our top 10 list.

To overcome this, we are going to create a really simple calculated field, called *Earthquake Counter*, which we will just set to 1 in the calculation. We can then add this to our values area to see the number of earthquakes by each region. To see these numbers as a percentage, we will navigate to the "Percentage Of" option in our Analysis menu and select "Column", before formatting to 0 decimal places and, once again, sorting by descending values for easier viewing.

At this point, we will, again, add our single date filter and add a title to our values column, following the same steps described in the section above, but on this occasion we will also use the "Edit Alias" option to shorten our column name to a slightly simpler description.

![alt text](/img/posts/earthquake-location-percentage.jpg "Table of Percentage of Earthquakes by Location Broad")

<br>

#### Frequency, Average Magnitude, and Maximum Magnitude of Earthquakes by Location

For our third feature, we will be creating a bar chart. However, rather than building 3 separate charts to add to our dashboard, we will be combining these into a single element.

To start, let's take a look at the frequency of earthquakes by location. We will first add the *Location* variable to our Rows section. We can then reuse our *Earthquake Counter* calculated field, dragging this onto our Values section. At this point we have the right data, but we still just have a table of data. To change this into a bar chart, we will first update the values to a count, rather than a sum, before using the "Show Me" options to select a horizontal bar chart.

We now want to make this a little easier to read. Firstly, we will again sort by descending values. Next, we will also add some labels to our chart by dragging the *Earthquake Counter* variable onto the Label icon in the Marks section. For consistency, we will again update this to a count.

Now let's move on to the average and maximum magnitude visuals. As we've already set the foundations of a chart by location, we can build on this to show all three of these data aggregations within a single element.

To add the average magnitude visual, we will just drag the *Magnitude* variable over to the Columns section and, straight away, we see a second bar chart showing values for the same list of locations. However, there are still a few steps to create an accurate visual.

Firstly, we need to update the measure to be an average. Secondly, we need to adjust the labels to match as these are still displaying as a sum of the *Earthquake Counter* variable. To change this, we will navigate back to our Marks section, making sure to select the appropriate tab for the average magnitude chart. We can then remove the sum of *Earthquake Counter* labels, and by using CTRL we can drag our *Average Magnitude* column variable and duplicate as a label. We may then adjust the formatting of these labels to whole numbers to stay consistent.

To add our maximum magnitude, we can quickly follow the same steps we used for our average magnitude visual, with the only change being to update the measure to maximum instead of average. We will then round this off by adding a date filter by following the steps mentioned in the previous section. We will then have 3 new charts combined into a single worksheet!

![alt text](/img/posts/location-analysis.jpg "Chart of Frequency, Average Magnitude, and Maximum Magnitude by Location")

<br>

#### Map of Earthquake Locations and their Intensity

For this visual, we're going to go a step further than data tables and bar charts, and build an interactive map.

To start us off, we are once again going to drag our relevant variables onto a new worksheet. We will drag *Latitude* onto the Rows section, and *Longitude* onto the Columns section. Now at this point, we only see a single average position of all our data. To map out each earthquake, we can just drag our *Id* variable onto the Detail icon of the Marks section and, as simple as that, we already have a global map showing the position of every earthquake in our dataset.

At this stage, we have a map of earthquakes by location, but we can't see any difference in intensity. To add this in, we can drag the *Magnitude* variable onto the Size icon of the Marks tab, and we will see that each point on the map has increased or decreased in size relative to the magnitude of the earthquake it represents.

However, the points on the map are still a little hard to differentiate, so it could be useful here to also add a colour gradient, in addition to the proportional sizing. To do this, we can again just drag the *Magnitude* variable onto the Marks section, but this time onto the Colour icon. By clicking on this Colour icon, we can then adjust these colours to whatever palette best tells the story of our visualisation. From this same location, we can also add borders to our earthquake location markers for further clarity between data points.

For the next step on this visualisation, let's first improve the interactivity of our map. There are already a number of data labels present in the tooltip for each earthquake plotted on the map. However, for some more obscure regions, or for when the map is very zoomed out, it may also be useful to add a location to help the user's understanding. We can do this very quickly by dragging the *Location* variable onto the Tooltip icon in the Marks section. Finally, we will, once again, add our date filter using the same steps we followed when building our initial chart.

![alt text](/img/posts/tableau-map-image.png "Map of Earthquake Locations and their Intensity")

<br>
<br>

___

# Creating The Dashboard  <a name="create-dashboard"></a>

We have now built all our individual elements to meet the requirements of our client. Our final step is to pull these all into a single dashboard and ensure it is visually appealing.

To start, we will create a new dashboard from the taskbar at the bottom of the page. We can then see each of our worksheets listed on the left-hand side, each ready to be dragged into place. Before we do this, let's give our dashboard a title, and to do so we can just give our tab a name and then check the "Show dashboard title" box in the bottom left of the page. We can then format this and move it to the appropriate position.

We are now ready to add our worksheets to the dashboard, and we will ensure that the "Tiled" layout is selected in the bottom left. This will ensure that our charts snap nicely into position without the need to spend lots of time lining things up. We can then drag each chart into the desired location on the dashboard. It is worth noting at this point, that each chart will bring across its own filters and legends to the panel on the right-hand side, so we can also remove any of these that aren't required.

The one filter we know to be key is our date filter, and we can now look to apply a single version of this to all of our charts. We can do this by choosing the "Apply to Worksheets" option from the More Options menu for the relevant filter and selecting all the worksheets we have added to the dashboard. Once this is complete, we can also look to make better use of the space on our dashboard by, once again, navigating to the More Options menu, and selecting "Floating". This will allow us to move and resize our filter, and also increase the space provided to our charts, which was previously taken up by the filter panel on the right-hand side.

Now let's ensure it looks visually appealing. Firstly, we can right click in any empty space around our tables and charts and ensure it fits the "Entire View", or tile, we have placed this in. We can then look to adjust our colouring. In this case, we will change our dashboard and chart backgrounds to be black, ensure all text is white, and then further format and colour any titles, borders, grid lines, bars, etc, as desired. For some items, such as colouring the bars of our location analysis chart, we may need to go back to the source worksheet itself to adjust these.

To finish off our dashboard, and add a professional feel, we are also going to add a logo. We can do this by choosing the "Image" option from the Objects section in the bottom left of the page. Once we have selected the relevant file and opened in Tableau, we can make our final positional adjustments to complete our dashboard!

To view and interact with the completed dashboard, please navigate back up to the [Results](#overview-results) section in the Project Overview.

