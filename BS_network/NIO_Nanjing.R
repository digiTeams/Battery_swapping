################################################################################ 
################ Matching Optimization Results ########################## 
################################################################################ 
rm(list = ls())
#####import packages
library(readxl)
library(openxlsx)
library(gmodels)
library(permute)
library(lattice)
library(cluster)
library(vegan)
library(psych)
library(grid)
library(ggplot2)
library(cowplot)
library(scatterplot3d)
library(rgl)
library(parallel)
library(GGally)
library(car)
library(gcookbook)
library(gvlma)
library(magrittr)
library(reshape2)
library(plyr)
library(dplyr)
library(dplyr)
library(zoo)  # 用于计算移动平均
library(tidyverse)
library(lubridate)
library(ggthemes)
library(openxlsx)
library(export)
library(Cairo)
library(showtext)
library(readr)
library(gmodels)
library(leaflet)
library(leaflet.extras)
library(leafletZH)
library(leafletCN)
library(sf)
library(sp)
library(htmltools)
library(htmlwidgets)
library(webshot)


###working space
setwd('C:/OneDrive/Team/BS_network')



########################################################
######## drivers data ######## 
drivers_1017 <- data.frame(read.xlsx("Nanjing_drivers_20191017.xlsx"))
str(drivers_1017)

drivers_1018 <- data.frame(read.xlsx("Nanjing_drivers_20191018.xlsx"))
str(drivers_1018)

drivers_1019 <- data.frame(read.xlsx("Nanjing_drivers_20191019.xlsx"))
str(drivers_1019)


# 车辆ID	经度	纬度	速度	方向	状态	数据发送时间	数据接收时间	车型	
# 行政区号 具体道路位置	道路等级	城市	区县
names(drivers_1017) <- c('ID', 'status', 'longitude', 'latitude', 'addcode', 
                         'district', 'city', 'send_time', 'send_date', 
                         'send_hour', 'send_minute')
summary(drivers_1017)


names(drivers_1018) <- c('ID', 'status', 'longitude', 'latitude', 'addcode', 
                         'district', 'city', 'send_time', 'send_date', 
                         'send_hour', 'send_minute')
summary(drivers_1018)


names(drivers_1019) <- c('ID', 'status', 'longitude', 'latitude', 'addcode', 
                         'district', 'city', 'send_time', 'send_date', 
                         'send_hour', 'send_minute')
summary(drivers_1019)


##Add data
drivers_1017$city <- 'Nanjing'
drivers_1018$city <- 'Nanjing'
drivers_1019$city <- 'Nanjing'

drivers_1017$date <- '2019-10-17'
drivers_1018$date <- '2019-10-18'
drivers_1019$date <- '2019-10-19'


#玄武、秦淮、建邺、鼓楼、栖霞、雨花台、江宁、浦口、六合、溧水、高淳
drivers <- drivers_1017
translate <- function(text, data=drivers){
  ad <- switch(text,
               '玄武区' = 'Xuanwu',
               '秦淮区' = 'Qinhuai',
               '建邺区' = 'Jianye',
               '鼓楼区' = 'Gulou',
               '栖霞区' = 'Qixia',
               '雨花台区' = 'Yuhuatai',
               '江宁区' = 'Jiangning',
               '浦口区' = 'Pukou',
               '六合区' = 'Liuhe',
               '溧水区' = 'Lishui',
               '高淳区' = 'Gaochun'
  )
  return(ad)
}


####parallel computing
numcore=detectCores(logical = FALSE)
cl <- makeCluster(getOption('cl.cores', numcore))
clusterExport(cl, "drivers") ##pass the input parameters

Adds <- drivers_1017$district
x <- parSapply(cl, Adds, translate) ##a vector
rownames(x) <- NULL
drivers_1017$district <- x


Adds <- drivers_1018$district
x <- parSapply(cl, Adds, translate) ##a vector
rownames(x) <- NULL
drivers_1018$district <- x


Adds <- drivers_1019$district
x <- parSapply(cl, Adds, translate) ##a vector
rownames(x) <- NULL
drivers_1019$district <- x

stopCluster(cl)



###analysis
by_id <- group_by(drivers_1017, ID)
summarise(by_id, n())

by_hr <- group_by(drivers_1017, hour)
summarise(by_hr, n())


by_id <- group_by(drivers_1019, ID)
summarise(by_id, n())

by_hr <- group_by(drivers_1019, hour)
summarise(by_hr, n())

drivers_1017$time[1:100]
drivers_1017$date[1:100]

drivers_1018$time[1:100]
drivers_1018$date[1:100]

drivers_1019$time[1:100]
drivers_1019$date[1:100]


##Save the processed data
drivers <- rbind(drivers_1017, drivers_1018, drivers_1019)
summary(drivers)
#write.csv(drivers, 'Nanjing_drivers_201910.csv', row.names = FALSE)

nrow(drivers)

x <- unique(drivers$ID)
length(x)







######## NIO BS network ######## 
BS_NJ <- data.frame(read.xlsx("Nanjing_NIO_2025-01-07.xlsx")) 
str(BS_NJ)


x <- c(1:nrow(BS_NJ))
ID <- paste('NIO', as.character(x), sep='-')

##get GPS coordinates
longitude <- vector(mode="numeric", length=nrow(BS_NJ))
latitude <- vector(mode="numeric", length=nrow(BS_NJ))
district <- vector(mode="character", length=nrow(BS_NJ))

for (k in 1:nrow(BS_NJ)) {
  loc <- BS_NJ$location[k]
  x <- unlist(strsplit(loc, ','))
  longitude[k] <- as.numeric(x[1])
  latitude[k] <- as.numeric(x[2])
  
  add <- BS_NJ$district[k]
  district[k] <- translate(add, drivers)
}

city <- rep('Nanjing', times = nrow(BS_NJ))


NIO <- data.frame(ID, longitude, latitude, district, city)
str(NIO)
write.csv(NIO, 'Nanjing_NIO_202501.csv', row.names = FALSE)






######## The city network ########
Drivers <- data.frame(read.csv("Nanjing_drivers_2025.csv")) 
str(Drivers)

Clusters <- data.frame(read.csv("Nanjing_clusters_2025.csv")) 
str(Clusters)

Stations <- data.frame(read.csv("Nanjing_stations_2025.csv")) 
str(Stations)

Requests <- data.frame(read.csv("Request_data_sample.csv")) 
str(Requests)



###### Data points ######
# "id"  "cluster"  "area"  "longitude"  "latitude"  "submission" "SOC" 
type <- rep('request', times = nrow(Requests))
X1 <- Requests[,c('id', 'longitude', 'latitude')] %>% mutate(type)

# "id"  "cluster"  "longitude"  "latitude"   "battery_cap" "chargers"  "charge_time"  "swap_time"  
type <- rep('station', times = nrow(Stations))
X2 <- Stations[,c('id', 'longitude', 'latitude')] %>% mutate(type)
X2

Points <- rbind(X1, X2)
Points$type <- factor(Points$type, order=TRUE, 
                      levels=c('station', 'request'))
summary(Points)


###### Cluster polygons ######
str(Clusters)

ind <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
         17, 18, 19, 20, 24, 25, 26, 27, 31, 32, 33, 34, 37, 
         38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48) + 1
Clusters$type[ind] <- c("suburb")
ind <- c(16, 21, 22, 23, 28, 29, 30, 35, 36) + 1
Clusters$type[ind] <- c("centre")

Clusters$type

parse_boundary_points <- function(boundary_str) {
  # 移除字符串开头和结尾的方括号
  cleaned_str <- gsub("^\\[|\\]$", "", boundary_str)
  # 分割成单个坐标点
  points <- strsplit(cleaned_str, "\\), \\(")[[1]]
  # 清理每个坐标点并转换为数值矩???
  coords <- t(sapply(points, function(point) {
    clean_point <- gsub("[\\(\\)]", "", point)
    as.numeric(strsplit(clean_point, ", ")[[1]])
  }))
  return(coords)
}



###### City base map ######
nanjing_sf <- st_read("Nanjing.geoJson") # 读取GeoJSON

palx <- colorFactor(c("DodgerBlue", "Red"), 
                    domain = c('station', 'request'))

BS <- makeIcon(
  "BS.png",
  iconWidth = 15, iconHeight = 15,
  iconAnchorX = 5, iconAnchorY = 5
)


Map <- leaflet(nanjing_sf) %>%
  addTiles() %>%   
  amap(group = "Amap") %>%
  addProviderTiles("CartoDB.Positron") %>%
  addPolygons(
    color = "black", weight = 1.0, fillOpacity = 0.0 
  ) %>%
  addMarkers(data = Stations, ~longitude, ~latitude, icon = BS) %>%
  addCircleMarkers(data = Points, ~longitude, ~latitude, 
                   radius = 3.0, stroke = TRUE, color = ~palx(type), 
                   weight = 0.5, opacity = 1.0, fill = TRUE, 
                   fillColor = ~palx(type), fillOpacity = 1.0) %>%
  setView(lng=118.796624, lat=32.059344, zoom=10)

Map
#fillColor = "blue", fillOpacity = 0.1, 


###### Add clusters in Polygons ######
str(Clusters)
# 为每个多边形创建颜色，根据type列区 
type_pal <- colorFactor(
  palette = c("red", "blue"),
  domain = c("centre", "suburb")
)

for(i in 1:nrow(Clusters)) {
  cluster <- Clusters[i, ]
  
  # 解析边界 
  boundary_coords <- parse_boundary_points(cluster$boundary_points)
  
  # 添加多边 
  Map <- Map %>%
    addPolygons(
      lng = boundary_coords[, 1], lat = boundary_coords[, 2],
      fillColor = type_pal(cluster$type), fillOpacity = 0.05,
      weight = 0.1, stroke = TRUE
    ) %>%
    # 添加边界 
    addPolylines(
      lng = boundary_coords[, 1], lat = boundary_coords[, 2],
      color = "black", weight = 0.5, opacity = 1.0
    ) %>%
    addCircleMarkers(lng = as.numeric(gsub(".*\\((.*),.*", "\\1", cluster$centroid)),
                     lat = as.numeric(gsub(".*, (.*)\\).*", "\\1", cluster$centroid)),
                     radius = 10.0, stroke = TRUE, color = "orange", 
                     weight = 1.0, opacity = 0.1, fill = TRUE, 
                     fillColor = type_pal(cluster$type), fillOpacity = 0.1) %>%
    # 在中心添加ID标签
    addLabelOnlyMarkers(
      lng = as.numeric(gsub(".*\\((.*),.*", "\\1", cluster$centroid)),
      lat = as.numeric(gsub(".*, (.*)\\).*", "\\1", cluster$centroid)),
      label = as.character(cluster$id),
      labelOptions = labelOptions(
        noHide = TRUE, direction = "center", textOnly = TRUE,
        style = list(
          "color" = "black",
          "font-weight" = "bold",
          "font-size" = "12px"
        )
      )
    )
}


###### 添加图例 ######
#position = c("topright", "bottomright", "bottomleft", "topleft"),
Map <- Map %>%
  addLegend(
    data = Points, 
    position = c("topright"),
    pal = palx, values = ~type, opacity = 1.0, 
    title = "The types of points", 
    labels = c('Station', 'Request')
  ) %>%
  addLegend(
    data = Clusters,
    position = c("topright"),
    pal = type_pal, values = ~type, opacity = 0.1, 
    title = "Cluster Types",
    group = "Clusters"
  ) %>%
  setView(lng=118.796624, lat=32.059344, zoom=10)

Map



########################################################
######## Demand pattern ########
Requests <- data.frame(read.csv("Request_data_sample.csv")) 
str(Requests)

Requests$submission <- gsub(" LMT", "", Requests$submission)
Requests$submission <- as.POSIXct(Requests$submission, format = "%Y-%m-%d %H:%M:%OS")
Requests$hour <- as.numeric(format(Requests$submission, "%H"))
Requests$hour

head(Requests)
data <- Requests[c('id', 'hour', 'SOC')]
summary(data)

### Demand
fig1 <- ggplot(data, aes(x = hour)) +
  geom_bar(fill = "DodgerBlue", color = "black", alpha = 0.5, linewidth=0.5) +
  labs(x='hour', y='demand', title='(a) Distribution of hourly demand') + 
  scale_x_continuous(limits=c(6, 23), breaks=seq.int(6, 23, 1)) + 
  scale_y_continuous(breaks=seq.int(0, 200, 25)) +  
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(),
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.6,"cm"),
        legend.title=element_text(size=rel(1.15)),
        legend.text=element_text(size=rel(1.15)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.2)),
        axis.title.y = element_text(size=rel(1.2)),
        axis.text.x = element_text(size=rel(1.15)),
        axis.text.y = element_text(size=rel(1.15))) 
fig1



### SOC 
fig2 <- ggplot(data, aes(x = SOC)) +  
  geom_density(colour = "black", fill = "DodgerBlue", alpha = .5, linewidth=0.5) + 
  labs(x = 'SOC', y = "probability density", title='(b) Distribution of EV SOC') +
  scale_x_continuous(limits=c(10, 20), breaks = seq.int(10, 20, 2.0)) +
  scale_y_continuous(breaks = seq.int(0, 1, 0.025)) +  
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.6,"cm"),
        legend.title=element_text(size=rel(1.15)),
        legend.text=element_text(size=rel(1.15)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.2)),
        axis.title.y = element_text(size=rel(1.2)),
        axis.text.x = element_text(size=rel(1.15)),
        axis.text.y = element_text(size=rel(1.15))) 
fig2


##export: Figure-demand, 5 x 12
library(grid)
grid.newpage()
pushViewport(viewport(layout = grid.layout(1,2)))
vplayout <- function(x,y) {
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(fig1, vp = vplayout(1,1))
print(fig2, vp = vplayout(1,2))





########################################################
rm(list = ls())
#####import packages
library(readxl)
library(openxlsx)
library(gmodels)
library(permute)
library(lattice)
library(cluster)
library(vegan)
library(psych)
library(grid)
library(ggplot2)
library(cowplot)
library(scatterplot3d)
library(rgl)
library(parallel)
library(GGally)
library(car)
library(gcookbook)
library(gvlma)
library(magrittr)
library(reshape2)
library(plyr)
library(dplyr)
library(dplyr)
library(zoo)  # 用于计算移动平均
library(tidyverse)
library(lubridate)
library(ggthemes)
library(openxlsx)
library(export)
library(Cairo)
library(showtext)
library(readr)
library(gmodels)
library(leaflet)
library(leaflet.extras)
library(leafletZH)
library(leafletCN)
library(sf)
library(sp)
library(htmltools)
library(htmlwidgets)
library(webshot)


###working space
setwd('C:/OneDrive/Team/BS_network')

########################################################
######## Training dataset ######## 
Instances <- data.frame(read.csv("Instances_summary_11-20.csv"))
Instances <- Instances[-1,]
str(Instances)

optDRL <- data.frame(read.csv("optDRL_policy_11-20.csv"))
#optDRL$total_profits <- optDRL$total_revenues - optDRL$total_subsidies
optDRL <- optDRL[-1,]
str(optDRL)

w4DRL <- data.frame(read.csv("w4DRL_policy_11-20.csv"))
#w42DRL$total_profits <- w4DRL$total_revenues - w4DRL$total_subsidies
w4DRL <- w4DRL[-1,]
str(w4DRL)

w2DRL <- data.frame(read.csv("w2DRL_policy_11-20.csv"))
w2DRL$total_profits <- w2DRL$total_revenues - w2DRL$total_subsidies
w2DRL <- w2DRL[-1,]
str(w2DRL)

fairDRL <- data.frame(read.csv("fairDRL_policy_11-20.csv"))
fairDRL$total_profits <- fairDRL$total_revenues - fairDRL$total_subsidies
fairDRL <- fairDRL[-1,]
str(fairDRL)

Greedy <- data.frame(read.csv("fairGreedy_policy_11-20.csv"))
Greedy$total_profits <- Greedy$total_revenues - Greedy$total_subsidies
Greedy <- Greedy[-1,]
str(Greedy)

RHO <- data.frame(read.csv("fairRHO_policy_11-20.csv"))
RHO$total_profits <- RHO$total_revenues - RHO$total_subsidies
RHO <- RHO[-1,]
str(RHO)


# 计算移动平均和置信区
window_size <- 32  # 设置移动平均窗口大小
z <- 1.96  #90%: 1.645; 95%: 1.96

calstat <- function(data, colname, window_size, z, scale){
  episode <- data$episode
  demand <- data$total_demand
  
  values <- data[[colname]]
  n <- length(values)
  
  if (scale > 1) {
    for(i in 1:n) {
      values[i] <- values[i] / demand[i]
    }
  }
  
  # 初始化结果向
  moving_avg <- numeric(n)
  moving_sd <- numeric(n)
  
  # 计算每个点的移动平均和移动标准差
  for(i in 1:n) {
    if(i < window_size) {
      # k < window_size 时，使用1到k的所有数
      window_values <- values[1:i]
    } else {
      # k >= window_size 时，使用标准的移动窗
      window_values <- values[(i-window_size+1):i]
    }
    
    moving_avg[i] <- mean(window_values)
    moving_sd[i] <- sd(window_values)
  }
  
  # 计算实际窗口大小（用于标准误差计算）
  actual_window_size <- pmin(1:n, window_size)
  
  # 计算标准误差
  se <- moving_sd / sqrt(actual_window_size)
  
  # 计算置信区间
  ci_lower <- moving_avg - z * se
  ci_upper <- moving_avg + z * se
  
  result <- data.frame(episode, moving_avg, ci_lower, ci_upper)
  names(result) <- c('episode', 'moving_avg', 'ci_lower', 'ci_upper')
  return(result)
}



########## Training data #####
summary(Instances)
str(Instances)
centrid <- 22  #centre cluster id
subid <- 10  #suburb cluster id

cc <- paste('cluster', centrid, sep = "_")
cs <- paste('cluster', subid, sep = "_")
data <- Instances[, c('episode', 'total_demand', cc, cs)]
summary(data)

ind <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
         17, 18, 19, 20, 24, 25, 26, 27, 31, 32, 33, 34, 37, 
         38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48)
suburbs <- paste('cluster', ind, sep = "_")
suburbs

suburb_demand <- numeric(nrow(Instances))
for (k in 1:nrow(Instances)) {
  x <- Instances[k, c(suburbs)]
  suburb_demand[k] <- sum(x)
}
suburb_demand

ind <- c(16, 21, 22, 23, 28, 29, 30, 35, 36)
centres <- paste('cluster', ind, sep = "_")
centres

centre_demand <- numeric(nrow(Instances))
for (k in 1:nrow(Instances)) {
  x <- Instances[k, c(centres)]
  centre_demand[k] <- sum(x)
}
centre_demand

episode <- Instances$episode
total_demand <- Instances$total_demand
data <- data.frame(episode, total_demand, suburb_demand, centre_demand)
summary(data)

c(mean(total_demand), sd(total_demand))
c(mean(suburb_demand), sd(suburb_demand))
c(mean(centre_demand), sd(centre_demand))

datX <- data.frame(episode, suburb_demand, centre_demand)
data <- melt(datX, id='episode') ## convert to long format
data$variable <- factor(data$variable, order=TRUE, levels=c('centre_demand', 'suburb_demand'))
summary(data)


#### Total demand
fig1 <- ggplot(Instances, aes(x = total_demand)) +  
  geom_density(fill = "DodgerBlue", colour = "black", alpha = .5, linewidth=0.5) + 
  labs(x = 'total demand', y = 'probability density', 
       title='(a) Distribution of total demand') +
  scale_x_continuous(breaks = seq.int(1000, 3000, 50)) +
  scale_y_continuous(breaks = seq.int(0, 1, 0.0025)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.01, 0.99), legend.justification=c(0.01, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.3,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 

fig1


#### Cluster demand
#"OrangeRed", "DodgerBlue"
fig2 <- ggplot(data, aes(x = value, fill = variable)) +  
  geom_density(colour = "black", alpha = .5, linewidth=0.5) + 
  scale_fill_manual(labels = c('centre demand', 'suburb demand'), 
                    values = c("OrangeRed", "DodgerBlue")) + 
  labs(x = 'area demand', y = 'probability density', 
       title='(b) Distributions of area demand', fill = "Types of demand") +
  scale_x_continuous(breaks = seq.int(0, 2000, 50)) +
  scale_y_continuous(breaks = seq.int(0, 0.015, 0.0025)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.99), legend.justification=c(0.99, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 

fig2


##export: Figure-instances, 5 x 12
library(grid)
grid.newpage()
pushViewport(viewport(layout = grid.layout(1,2)))
vplayout <- function(x,y) {
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(fig1, vp = vplayout(1,1))
print(fig2, vp = vplayout(1,2))




#################################################### 
############# Training performance ################# 
str(fairDRL)

##efficiency comparision
# x <- (optDRL$total_profits[1500:3000] - optDRL$total_profits[1500:3000])/optDRL$total_profits[1500:3000]
# x
# mean(x)
x <- (optDRL$total_profits[1500:3000] - fairDRL$total_profits[1500:3000])/fairDRL$total_profits[1500:3000]
x
mean(x)

x <- (w2DRL$total_profits[1500:3000] - fairDRL$total_profits[1500:3000])/fairDRL$total_profits[1500:3000]
x
mean(x)

x <- (fairDRL$total_profits[1500:3000] - RHO$total_profits[1500:3000])/RHO$total_profits[1500:3000]
x
c(mean(x), min(x), max(x))

x <- (fairDRL$total_profits[1500:3000] - Greedy$total_profits[1500:3000])/Greedy$total_profits[1500:3000]
x
c(mean(x), min(x), max(x))




### Fair reward curves ###
scale = 1e4

datD <- calstat(fairDRL, 'total_rewards', window_size, z, scale)
datD$case <- c("fairDRL")

datG <- calstat(Greedy, 'total_rewards', window_size, z, scale)
datG$case <- c("Greedy")

datR <- calstat(RHO, 'total_rewards', window_size, z, scale)
datR$case <- c("RHO")

data <- rbind(datD, datR, datG)
data$case <- factor(data$case, order=TRUE, 
                    levels=c("fairDRL", "RHO", "Greedy"))

summary(data)


fig1 <- ggplot(data, aes(x = episode, y = moving_avg, colour = case)) + 
  geom_line(size=0.5) + geom_point(size=0.5) + 
  geom_ribbon(aes(x = episode, ymin = ci_lower, ymax = ci_upper, 
                  fill = case), alpha = 0.2, linewidth = 0.2, 
              show.legend = FALSE) + 
  labs(x='episode', y='rewards (unit: CNY)', 
       title='(a) Normalized reward curves of fair policies',
       fill='Policy', color='Policy') +
  scale_color_manual(values = c("OrangeRed", "DodgerBlue", "Black"),
                     labels = c("fairDRL", "RHO", "Greedy")) + 
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue", "Black"),
                    labels = c("fairDRL", "RHO", "Greedy")) + 
  scale_x_continuous(limits=c(0, 3000), breaks=seq.int(0, 3000, 250)) +
  scale_y_continuous(breaks=seq.int(20, 40, 1.0)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.01), legend.justification=c(0.99, 0.01), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig1


### Unfair reward curves ###
scale = 1e4

datO <- calstat(optDRL, 'total_rewards', window_size, z, scale)
datO$case <- c("optDRL")

datW2 <- calstat(w2DRL, 'total_rewards', window_size, z, scale)
datW2$case <- c("w2DRL")

datW4 <- calstat(w4DRL, 'total_rewards', window_size, z, scale)
datW4$case <- c("w4DRL")

data <- rbind(datO, datW2, datW4)
data$case <- factor(data$case, order=TRUE, 
                    levels=c("optDRL", "w2DRL", "w4DRL"))

summary(data)


fig2 <- ggplot(data, aes(x = episode, y = moving_avg, colour = case)) + 
  geom_line(size=0.5) + geom_point(size=0.5) + 
  geom_ribbon(aes(x = episode, ymin = ci_lower, ymax = ci_upper, 
                  fill = case), alpha = 0.2, linewidth = 0.2, 
              show.legend = FALSE) + 
  labs(x='episode', y='rewards (unit: CNY)', 
       title='(b) Normalized reward curves of unfair policies',
       fill='Policy', color='Policy') +
  scale_color_manual(values = c("Purple", "SeaGreen", "Orange"),
                     labels = c("optDRL", "w2DRL", "w4DRL")) + 
  scale_fill_manual(values = c("Purple", "SeaGreen", "Orange"),
                    labels = c("optDRL", "w2DRL", "w4DRL")) + 
  scale_x_continuous(limits=c(0, 3000), breaks=seq.int(0, 3000, 250)) +
  scale_y_continuous(breaks=seq.int(20, 40, 1.0)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.01), legend.justification=c(0.99, 0.01), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig2


###### Optimization profits ######
scale = 1e4

datD <- calstat(fairDRL, 'total_profits', window_size, z, scale)
datD$case <- c("fairDRL")

datR <- calstat(RHO, 'total_profits', window_size, z, scale)
datR$case <- c("RHO")

datG <- calstat(Greedy, 'total_profits', window_size, z, scale)
datG$case <- c("Greedy")

datO <- calstat(optDRL, 'total_profits', window_size, z, scale)
datO$case <- c("optDRL")

datW2 <- calstat(w2DRL, 'total_profits', window_size, z, scale)
datW2$case <- c("w2DRL")

datW4 <- calstat(w4DRL, 'total_profits', window_size, z, scale)
datW4$case <- c("w4DRL")

data <- rbind(datD, datR, datG, datO, datW2, datW4)
data$case <- factor(data$case, order=TRUE, 
                    levels=c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL"))

summary(data)


fig3 <- ggplot(data, aes(x = episode, y = moving_avg, colour = case)) + 
  geom_line(size=0.5) + geom_point(size=0.5) + 
  geom_ribbon(aes(x = episode, ymin = ci_lower, ymax = ci_upper, 
                  fill = case), alpha = 0.2, linewidth = 0.2, 
              show.legend = FALSE) + 
  labs(x='episode', y='profits (unit: CNY)', 
       title='(c) Normalized profit curves of policies',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("OrangeRed", "DodgerBlue", "Black", "Purple", "SeaGreen", "Orange"),
                     labels = c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")) + 
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue", "Black", "Purple", "SeaGreen", "Orange"),
                    labels = c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")) + 
  scale_x_continuous(limits=c(0, 3000), breaks=seq.int(0, 3000, 250)) +
  scale_y_continuous(breaks=seq.int(53, 65, 1.0)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.01), legend.justification=c(0.99, 0.01), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.25,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig3


###### Training loss ######
str(fairDRL)
scale = 1.0

datD <- calstat(fairDRL, 'loss', window_size, z, scale)
datD$case <- c("fairDRL")

datO <- calstat(optDRL, 'loss', window_size, z, scale)
datO$case <- c("optDRL")

datW2 <- calstat(w2DRL, 'loss', window_size, z, scale)
datW2$case <- c("w2DRL")

datW4 <- calstat(w4DRL, 'loss', window_size, z, scale)
datW4$case <- c("w4DRL")


data <- rbind(datD, datO, datW2, datW4)
data$case <- factor(data$case, order=TRUE, 
                    levels=c("fairDRL", "optDRL", "w2DRL", "w4DRL"))

#data <- data[c(1001:2000),]
summary(data)


fig4 <- ggplot(data, aes(x = episode, y = moving_avg, colour = case)) + 
  geom_line(size=0.5) + geom_point(size=0.5) + 
  geom_ribbon(aes(x = episode, ymin = ci_lower, ymax = ci_upper, 
                  fill = case), alpha = 0.2, linewidth = 0.2, 
              show.legend = FALSE) + 
  labs(x='episode', y='MSE loss', title='(d) Training loss curves of DRL policies',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("OrangeRed", "Purple", "SeaGreen", "Orange"),
                     labels = c("fairDRL", "optDRL", "w2DRL", "w4DRL")) + 
  scale_fill_manual(values = c("OrangeRed", "Purple", "SeaGreen", "Orange"),
                    labels = c("fairDRL", "optDRL", "w2DRL", "w4DRL")) + 
  scale_x_continuous(limits=c(0, 3000), breaks=seq.int(0, 3000, 250)) +
  scale_y_continuous(limits=c(500, 6000), breaks=seq.int(500, 6000, 1000)) +
  theme_bw() +  
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.99), legend.justification=c(0.99, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig4


##export: Figure-training, 8 x 10
library(grid)
grid.newpage()
pushViewport(viewport(layout = grid.layout(2,2)))
vplayout <- function(x,y) {
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(fig1, vp = vplayout(1,1))
print(fig2, vp = vplayout(1,2))
print(fig3, vp = vplayout(2,1))
print(fig4, vp = vplayout(2,2))




##############################################
#['episode', 'total_demand', 'served', 'charged', 'rejection', 
#  'total_rewards', 'total_revenues', 'total_subsidies', 'total_envies',
#  'max_wait', 'mean_wait', 'std_wait', 'max_detour', 'mean_detour', 'std_detour',
#  'max_subsidy', 'mean_subsidy', 'std_subsidy', 'max_envy', 'mean_envy', 'std_envy', 
#  'max_queue', 'mean_queue', 'std_queue',  'avg_station_serve', 'std_station_serve', 
#  'avg_station_charge', 'std_station_charge', 'avg_cls_envy', 'std_cls_envy', 
#  ... ... (envies in clusters)
#  'run_time', 'loss']

str(fairDRL)

z <- 1.0  #90%: 1.645; 95%: 1.96
calinsts <- function(episode, means, sds, maxs, z){
  n <- length(episode)
  ci_lower <- numeric(n)
  ci_upper <- numeric(n)
  
  # 计算每个点的移动平均和移动标准差
  for(i in 1:n) {
    ci_lower[i] <- max(means[i] - z * sds, 0)
    ci_upper[i] <- min(means[i] + z * sds, maxs[i])
  }
  
  result <- data.frame(episode, means, ci_lower, ci_upper)
  names(result) <- c('episode', 'mean', 'ci_lower', 'ci_upper')
  return(result)
}



#### Request subsidy variations ####
episode <- fairDRL$episode
means <- fairDRL$mean_subsidy
sds <- fairDRL$std_subsidy
maxs <- fairDRL$max_subsidy

datD <- calinsts(episode, means, sds, maxs, z)
datD$case <- c("fairDRL")

names(datD)
summary(datD)


ggplot(datD, aes(x = episode, y = mean)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
              alpha = 0.25, fill = "#4A90E2") +
  geom_line(color = "#4A90E2", size = 0.5) +
  geom_point(color = "#4A90E2", size = 0.5) +
  labs(x = "Episode", y = "Mean Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank()
  )




#### Queuing time ####
str(fairDRL)
window_size <- 32
z <- 1.96  #90%: 1.645; 95%: 1.96
scale = 1.0


data <- calstat(fairDRL, 'mean_wait', window_size, z, scale)
data$case <- c("fairDRL")
str(data)

ggplot(data, aes(x = episode, y = moving_avg)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
              alpha = 0.5, fill = "#4A90E2") +
  geom_line(color = "#4A90E2", size = 0.5) +
  geom_point(color = "#4A90E2", size = 0.5) +
  labs(x = "episode", y = "rquest waiting time") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank()
  )



#### Detour distance ####
data <- calstat(fairDRL, 'mean_detour', window_size, z, scale)
data$case <- c("fairDRL")
str(data)

ggplot(data, aes(x = episode, y = moving_avg)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
              alpha = 0.5, fill = "#4A90E2") +
  geom_line(color = "#4A90E2", size = 0.5) +
  geom_point(color = "#4A90E2", size = 0.5) +
  labs(x = "episode", y = "rquest detour time") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank()
  )



#### Cluster envy ####
data <- calstat(fairDRL, 'avg_cls_envy', window_size, z, scale)
data$case <- c("fairDRL")
str(data)

ggplot(data, aes(x = episode, y = moving_avg)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
              alpha = 0.5, fill = "#4A90E2") +
  geom_line(color = "#4A90E2", size = 0.5) +
  geom_point(color = "#4A90E2", size = 0.5) +
  labs(x = "episode", y = "cluster envy") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank()
  )


#### Efficiency loss ####
str(fairDRL)
xF <- fairDRL$total_profits
xO <- w2DRL$total_profits

loss <- 100 * (xO - xF) / xF
episode <- fairDRL$episode
data <- data.frame(episode, loss)
summary(data)
summary(loss[1500:2000])

ggplot(data, aes(x = episode, y = loss)) +
  geom_line(color = "#4A90E2", size = 0.5) +
  geom_point(color = "#4A90E2", size = 0.5) +
  labs(x = "episode", y = "efficiency loss") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank()
  )




#################################################################### 
################### Service level and fairness #####################
### Request subsidy ###
str(fairDRL)
z <- 1.96  #90%: 1.645; 95%: 1.96
scale = 1.0

summary(RHO$mean_subsidy)
summary(Greedy$mean_subsidy)
summary(fairDRL$mean_subsidy)


datD <- calstat(fairDRL, 'mean_subsidy', window_size, z, scale)
datD$case <- c("fairDRL")

datR <- calstat(RHO, 'mean_subsidy', window_size, z, scale)
datR$case <- c("RHO")
# datR$ci_lower <- datR$moving_avg
# datR$ci_upper <- datR$moving_avg

datG <- calstat(Greedy, 'mean_subsidy', window_size, z, scale)
datG$case <- c("Greedy")

data <- rbind(datD, datR, datG)
data$case <- factor(data$case, order=TRUE, 
                    levels=c("fairDRL", "RHO", "Greedy"))

summary(data)


fig1 <- ggplot(data, aes(x = episode, y = moving_avg, colour = case)) + 
  geom_line(size=0.5) + geom_point(size=0.5) + 
  geom_ribbon(aes(x = episode, ymin = ci_lower, ymax = ci_upper, 
                  fill = case), alpha = 0.2, linewidth = 0.2, 
              show.legend = FALSE) + 
  labs(x='episode', y='compensation (unit: CNY)', 
       title='(a) Average compensations of fair policies',
       fill='Policy', color='Policy') +
  scale_color_manual(values = c("OrangeRed", "DodgerBlue", "Black"),
                     labels = c("fairDRL", "RHO", "Greedy")) +
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue", "Black"),
                    labels = c("fairDRL", "RHO", "Greedy")) +
  scale_x_continuous(limits=c(0, 3000), breaks=seq.int(0, 3000, 250)) +
  scale_y_continuous(breaks=seq.int(0, 5, 0.5)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.99), legend.justification=c(0.99, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig1



###### Request envies ######
summary(Greedy)

z <- 1.96  #90%: 1.645; 95%: 1.96
scale = 1.0

datO <- calstat(optDRL, 'mean_envy', window_size, z, scale)
datO$case <- c("optDRL")

datW2 <- calstat(w2DRL, 'mean_envy', window_size, z, scale)
datW2$case <- c("w2DRL")

datW4 <- calstat(w4DRL, 'mean_envy', window_size, z, scale)
datW4$case <- c("w4DRL")


data <- rbind(datO, datW2, datW4)
data$case <- factor(data$case, order=TRUE, 
                    levels=c("optDRL", "w2DRL", "w4DRL"))

summary(data)


fig2 <- ggplot(data, aes(x = episode, y = moving_avg, colour = case)) + 
  geom_line(size=0.5) + geom_point(size=0.5) + 
  geom_ribbon(aes(x = episode, ymin = ci_lower, ymax = ci_upper, 
                  fill = case), alpha = 0.2, linewidth = 0.2, 
              show.legend = FALSE) + 
  labs(x='episode', y='envy (unit: CNY)', 
       title='(b) Average request envies of unfair policies',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("Purple", "SeaGreen", "Orange"),
                     labels = c("optDRL", "w2DRL", "w4DRL")) + 
  scale_fill_manual(values = c("Purple", "SeaGreen", "Orange"),
                    labels = c("optDRL", "w2DRL", "w4DRL")) + 
  scale_x_continuous(limits=c(0, 3000), breaks=seq.int(0, 3000, 250)) +
  scale_y_continuous(breaks=seq.int(0, 4, 0.2)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.99), legend.justification=c(0.99, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig2



###################################################
### Request waiting time ###
summary(fairDRL)
window_size <- 32
z <- 1.96  #90%: 1.645; 95%: 1.96
scale = 1.0

datD <- calstat(fairDRL, 'mean_wait', window_size, z, scale)
datD$case <- c("fairDRL")

datR <- calstat(RHO, 'mean_wait', window_size, z, scale)
datR$case <- c("RHO")

datG <- calstat(Greedy, 'mean_wait', window_size, z, scale)
datG$case <- c("Greedy")

datO <- calstat(optDRL, 'mean_wait', window_size, z, scale)
datO$case <- c("optDRL")

datW2 <- calstat(w2DRL, 'mean_wait', window_size, z, scale)
datW2$case <- c("w2DRL")

datW4 <- calstat(w4DRL, 'mean_wait', window_size, z, scale)
datW4$case <- c("w4DRL")


data <- rbind(datD, datR, datG, datO, datW2, datW4)
data$case <- factor(data$case, order=TRUE, 
                    levels=c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL"))

summary(data)


fig3 <- ggplot(data, aes(x = episode, y = moving_avg, colour = case)) + 
  geom_line(size=0.5) + geom_point(size=0.5) + 
  geom_ribbon(aes(x = episode, ymin = ci_lower, ymax = ci_upper, 
                  fill = case), alpha = 0.2, linewidth = 0.2, 
              show.legend = FALSE) + 
  labs(x='episode', y='waiting time (unit: minute)', 
       title='(c) Average waiting time of policies',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("OrangeRed", "DodgerBlue", "Black", "Purple", "SeaGreen", "Orange"),
                     labels = c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")) + 
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue", "Black", "Purple", "SeaGreen", "Orange"),
                    labels = c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")) +
  scale_x_continuous(limits=c(0, 3000), breaks=seq.int(0, 3000, 250)) +
  scale_y_continuous(breaks=seq.int(0, 10, 0.5)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.01, 0.99), legend.justification=c(0.01, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.25,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig3


### Request detour distance ###
str(fairDRL)
window_size <- 32
z <- 1.96  #90%: 1.645; 95%: 1.96
scale = 1.0

datD <- calstat(fairDRL, 'mean_detour', window_size, z, scale)
datD$case <- c("fairDRL")

datR <- calstat(RHO, 'mean_detour', window_size, z, scale)
datR$case <- c("RHO")

datG <- calstat(Greedy, 'mean_detour', window_size, z, scale)
datG$case <- c("Greedy")

datO <- calstat(optDRL, 'mean_detour', window_size, z, scale)
datO$case <- c("optDRL")

datW2 <- calstat(w2DRL, 'mean_detour', window_size, z, scale)
datW2$case <- c("w2DRL")

datW4 <- calstat(w4DRL, 'mean_detour', window_size, z, scale)
datW4$case <- c("w4DRL")

data <- rbind(datD, datR, datG, datO, datW2, datW4)
data$case <- factor(data$case, order=TRUE, 
                    levels=c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL"))

summary(data)


fig4 <- ggplot(data, aes(x = episode, y = moving_avg, colour = case)) + 
  geom_line(size=0.5) + geom_point(size=0.5) + 
  geom_ribbon(aes(x = episode, ymin = ci_lower, ymax = ci_upper, 
                  fill = case), alpha = 0.2, linewidth = 0.2, 
              show.legend = FALSE) + 
  labs(x='episode', y='detour distance (unit: km)', 
       title='(d) Average detour distance of policies',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("OrangeRed", "DodgerBlue", "Black", "Purple", "SeaGreen", "Orange"),
                     labels = c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")) + 
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue", "Black", "Purple", "SeaGreen", "Orange"),
                    labels = c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")) +
  scale_x_continuous(limits=c(0, 3000), breaks=seq.int(0, 3000, 250)) +
  scale_y_continuous(breaks=seq.int(0, 5, 0.2)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.2), legend.justification=c(0.99, 0.2), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.25,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig4



##export: Figure-services, 8 x 10
library(grid)
grid.newpage()
pushViewport(viewport(layout = grid.layout(2,2)))
vplayout <- function(x,y) {
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(fig1, vp = vplayout(1,1))
print(fig2, vp = vplayout(1,2))
print(fig3, vp = vplayout(2,1))
print(fig4, vp = vplayout(2,2))






########################################################
rm(list = ls())
#####import packages
library(readxl)
library(openxlsx)
library(gmodels)
library(permute)
library(lattice)
library(cluster)
library(vegan)
library(psych)
library(grid)
library(ggplot2)
library(cowplot)
library(scatterplot3d)
library(rgl)
library(parallel)
library(GGally)
library(car)
library(gcookbook)
library(gvlma)
library(magrittr)
library(reshape2)
library(plyr)
library(dplyr)
library(dplyr)
library(zoo)  # 用于计算移动平均
library(tidyverse)
library(lubridate)
library(ggthemes)
library(openxlsx)
library(export)
library(Cairo)
library(showtext)
library(readr)
library(gmodels)
library(leaflet)
library(leaflet.extras)
library(leafletZH)
library(leafletCN)
library(sf)
library(sp)
library(htmltools)
library(htmlwidgets)
library(webshot)


###working space
setwd('C:/OneDrive/Team/BS_network')


##################################################################
############## Test results ##############
Instances <- data.frame(read.csv("Instances_test_11-28.csv"))
str(Instances)

optDRL <- data.frame(read.csv("optDRL_test_11-28.csv"))
str(optDRL)

w4DRL <- data.frame(read.csv("w4DRL_test_11-28.csv"))
str(w4DRL)

w2DRL <- data.frame(read.csv("w2DRL_test_11-28.csv"))
str(w2DRL)

fairDRL <- data.frame(read.csv("fairDRL_test_11-28.csv"))
str(fairDRL)

Greedy <- data.frame(read.csv("fairGreedy_test_11-28.csv"))
str(Greedy)
Greedy$total_profits <- Greedy$total_revenues - Greedy$total_subsidies

RHO <- data.frame(read.csv("fairRHO_test_11-28.csv"))
str(RHO)


###################### Request records ########################
optReq <- data.frame(read.csv("Request_records_optDRL.csv"))
str(optReq)

w4Req <- data.frame(read.csv("Request_records_w4DRL.csv"))
str(w4Req)

w2Req <- data.frame(read.csv("Request_records_w2DRL.csv"))
str(w2Req)

fairReq <- data.frame(read.csv("Request_records_fairDRL.csv"))
str(fairReq)

GreedyReq <- data.frame(read.csv("Request_records_greedy.csv"))
str(GreedyReq)

RHOReq <- data.frame(read.csv("Request_records_RHO.csv"))
str(RHOReq)



###############################################
summary(Instances)

mean(Instances$total_demand)
sd(Instances$total_demand)



###############################################
names(fairDRL)
summary(fairDRL)

# ['episode', 'total_demand', 'served', 'charged', 'rejection', 'total_rewards', 
#   'total_profits', 'total_revenues', 'total_subsidies', 'total_envies',
#   'max_wait', 'mean_wait', 'std_wait', 'max_detour', 'mean_detour', 'std_detour',
#   'max_envy', 'mean_envy', 'std_envy', 'max_subsidy', 'mean_subsidy', 'std_subsidy', 
#   'max_queue', 'mean_queue', 'std_queue', 
#   'avg_station_serve', 'std_station_serve', 'avg_station_charge', 'std_station_charge', 
#   'avg_cls_envy', 'std_cls_envy', 'avg_cls_subsidy', 'std_cls_subsidy',
#   'avg_centre_envy', 'std_centre_envy', 'avg_centre_subsidy', 'std_centre_subsidy',
#   'avg_suburb_envy', 'std_suburb_envy', 'avg_suburb_subsidy', 'std_suburb_subsidy',
#   ... ...
#   'run_time', 'loss']

caltest <- function(data, sn, policy){
  # # centre vs. suburb
  # ind <- c(16, 21, 22, 23, 28, 29, 30, 35, 36)
  # centres <- paste('cluster', ind, sep = "_")
  # xs <- numeric(nrow(data))
  # for (k in 1:nrow(data)) {
  #   x <- as.numeric(data[k, centres])  
  #   xs[k] <- mean(x)
  # }
  # 
  # ind <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
  #          17, 18, 19, 20, 24, 25, 26, 27, 31, 32, 33, 34, 37, 
  #          38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48)
  # suburbs <- paste('cluster', ind, sep = "_")
  # xb <- numeric(nrow(data))
  # for (k in 1:nrow(data)) {
  #   x <- as.numeric(data[k, suburbs])
  #   xb[k] <- mean(x)
  # }
  # 
  pos <- (data$scenario == sn)
  if (policy == 'Fair') {
    result <- c(mean(data$rejection[pos]), sd(data$rejection[pos]),
                mean(data$total_profits[pos]/data$total_demand[pos]), 
                sd(data$total_profits[pos]/data$total_demand[pos]),
                mean(data$mean_subsidy[pos]), mean(data$std_subsidy[pos]),
                mean(data$mean_wait[pos]), mean(data$std_wait[pos]), 
                mean(data$mean_detour[pos]), mean(data$std_detour[pos]),
                mean(data$mean_queue[pos]), mean(data$std_queue[pos]),
                mean(data$avg_station_serve[pos]), mean(data$std_station_serve[pos]),
                mean(data$avg_station_charge[pos]), mean(data$std_station_charge[pos]),
                mean(data$avg_centre_subsidy[pos]), mean(data$std_centre_subsidy[pos]),
                mean(data$avg_suburb_subsidy[pos]), mean(data$std_suburb_subsidy[pos]))
  } else {
    result <- c(mean(data$rejection[pos]), sd(data$rejection[pos]),
                mean(data$total_profits[pos]/data$total_demand[pos]), 
                sd(data$total_profits[pos]/data$total_demand[pos]),
                mean(data$mean_envy[pos]), mean(data$std_envy[pos]),
                mean(data$mean_wait[pos]), mean(data$std_wait[pos]), 
                mean(data$mean_detour[pos]), mean(data$std_detour[pos]),
                mean(data$mean_queue[pos]), mean(data$std_queue[pos]),
                mean(data$avg_station_serve[pos]), mean(data$std_station_serve[pos]),
                mean(data$avg_station_charge[pos]), mean(data$std_station_charge[pos]),
                mean(data$avg_centre_envy[pos]), mean(data$std_centre_envy[pos]),
                mean(data$avg_suburb_envy[pos]), mean(data$std_suburb_envy[pos]))
  }
  
  return(result)
}


#########################
names(fairDRL)
names(fairReq)

calService <- function(data, Req, sn, policy){
  posd <- (data$scenario == sn)
  posr <- (Req$scenario == sn)
  
  if (policy == 'Fair') {
    result <- c(mean(data$rejection[posd]),  
                mean(data$total_profits[posd]/data$total_demand[posd]), 
                mean(data$avg_station_serve[posd]), 
                mean(data$avg_station_charge[posd]), 
                mean(data$mean_queue[posd]), 
                mean(data$avg_centre_subsidy[posd]), 
                mean(data$avg_suburb_subsidy[posd]),
                mean(Req$subsidy[posr]),
                mean(Req$wait[posr]), 
                mean(Req$detour[posr]))
  } else {
    result <- c(mean(data$rejection[posd]),  
                mean(data$total_profits[posd]/data$total_demand[posd]), 
                mean(data$avg_station_serve[posd]), 
                mean(data$avg_station_charge[posd]), 
                mean(data$mean_queue[posd]), 
                mean(data$avg_centre_envy[posd]), 
                mean(data$avg_suburb_envy[posd]),
                mean(Req$envy[posr]),
                mean(Req$wait[posr]), 
                mean(Req$detour[posr]))
  }
  
  return(result)
}

### Summarization results
#fairDRL, w2DRL, Greedy, RHO 
sn = 0
drl <- calService(fairDRL, fairReq, sn, 'Fair')
rho <- calService(RHO, RHOReq, sn, 'Fair')
greedy <- calService(Greedy, GreedyReq, sn, 'Fair')
opt <- calService(optDRL, optReq, sn, 'Unfair')
w2gt <- calService(w2DRL, w2Req, sn, 'Unfair')
w4gt <- calService(w4DRL, w4Req, sn, 'Unfair')

x0 <- rbind(drl, rho, greedy, opt, w2gt, w4gt)
x0
round(x0,2)


##
sn = 1
drl <- calService(fairDRL, fairReq, sn, 'Fair')
rho <- calService(RHO, RHOReq, sn, 'Fair')
greedy <- calService(Greedy, GreedyReq, sn, 'Fair')
opt <- calService(optDRL, optReq, sn, 'Unfair')
w2gt <- calService(w2DRL, w2Req, sn, 'Unfair')
w4gt <- calService(w4DRL, w4Req, sn, 'Unfair')

x1 <- rbind(drl, rho, greedy, opt, w2gt, w4gt)
x1
round(x1,2)


##
sn = 2
drl <- calService(fairDRL, fairReq, sn, 'Fair')
rho <- calService(RHO, RHOReq, sn, 'Fair')
greedy <- calService(Greedy, GreedyReq, sn, 'Fair')
opt <- calService(optDRL, optReq, sn, 'Unfair')
w2gt <- calService(w2DRL, w2Req, sn, 'Unfair')
w4gt <- calService(w4DRL, w4Req, sn, 'Unfair')

x2 <- rbind(drl, rho, greedy, opt, w2gt, w4gt)
x2
round(x2,2)

####
round(rbind(x0, x1, x2),3)


### Profits: fairDRL, w2DRL, Greedy, RHO 
x <- (fairDRL$total_profits - Greedy$total_profits)/Greedy$total_profits
x
c(mean(x), sd(x))

x <- (fairDRL$total_profits - RHO$total_profits)/RHO$total_profits
x
c(mean(x), sd(x))

x <- (optDRL$total_profits - fairDRL$total_profits)/fairDRL$total_profits
x
c(mean(x), sd(x))


### Subsidies
(mean(RHOReq$subsidy) - mean(fairReq$subsidy)) / mean(fairReq$subsidy)

(mean(GreedyReq$subsidy) - mean(fairReq$subsidy)) / mean(fairReq$subsidy)




################# Boxplots ##################### 
names(fairDRL)
id <- c(1:nrow(fairDRL))
fair <- fairDRL$total_profits/fairDRL$total_demand
rho <- RHO$total_profits/RHO$total_demand
greedy <- Greedy$total_profits/Greedy$total_demand
opt <- optDRL$total_profits/optDRL$total_demand
w2gt <- w2DRL$total_profits/w2DRL$total_demand
w4gt <- w4DRL$total_profits/w4DRL$total_demand

X <- data.frame(id, fair, rho, greedy, opt, w2gt, w4gt)
names(X) <- c("id", "fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")

data <- data.frame(melt(X, id="id"))
case <- vector(mode="character", length=6*nrow(fairDRL))
case[1:(3*nrow(fairDRL))] <- 'Fair'
case[(3*nrow(fairDRL)+1):(6*nrow(fairDRL))] <- 'Unfair'

data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL"))
data$case <- factor(case, order=TRUE, 
                    levels=c('Fair', 'Unfair'))
data
summary(data)


fig1 <- ggplot(data, aes(x=variable, y=value, colour=case, fill=case)) + 
  stat_boxplot(geom="errorbar", width=0.25, size=0.6) + 
  geom_boxplot(alpha = 0.6, width=0.3, size=0.6, notch=FALSE, outlier.colour="red", 
               outlier.size=2.0) + 
  stat_summary(fun="mean", geom="point", shape=7, size=2.5, colour = "Black") +
  labs(x=NULL, y='profits (CNY)', 
       title='(a) Distributions of normalized total profits',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("OrangeRed", "DodgerBlue"),
                     labels = c("Fair", "Unfair")) + 
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue"),
                    labels = c("Fair", "Unfair")) + 
  guides(colour=guide_legend(title=NULL)) + 
  guides(fill=guide_legend(title=NULL)) + 
  scale_y_continuous(limits=c(42.5, 50), breaks=seq.int(42.5, 50, 2.5)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.01), legend.justification=c(0.99, 0.01), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig1


### Subsidies/envies: fairDRL, w2DRL, Greedy, RHO 
names(fairDRL)
id <- c(1:nrow(fairDRL))
fair <- fairDRL$mean_subsidy
rho <- RHO$mean_subsidy
greedy <- Greedy$mean_subsidy
opt <- optDRL$mean_envy
w2gt <- w2DRL$mean_envy
w4gt <- w4DRL$mean_envy


X <- data.frame(id, fair, rho, greedy, opt, w2gt, w4gt)
names(X) <- c("id", "fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")

data <- data.frame(melt(X, id="id"))
case <- vector(mode="character", length=6*nrow(fairDRL))
case[1:(3*nrow(fairDRL))] <- 'Fair'
case[(3*nrow(fairDRL)+1):(6*nrow(fairDRL))] <- 'Unfair'

data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL"))
data$case <- factor(case, order=TRUE, 
                    levels=c('Fair', 'Unfair'))
data
summary(data)


fig2 <- ggplot(data, aes(x=variable, y=value, colour=case, fill=case)) + 
  stat_boxplot(geom="errorbar", width=0.25, size=0.6) + 
  geom_boxplot(alpha = 0.6, width=0.3, size=0.6, notch=FALSE, outlier.colour="red", 
               outlier.size=2.0) + 
  stat_summary(fun="mean", geom="point", shape=7, size=2.5, colour = "Black") +
  labs(x=NULL, y='compensation/envy (CNY)', 
       title='(b) Distributions of average compensations/envies',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("OrangeRed", "DodgerBlue"),
                     labels = c("Fair", "Unfair")) + 
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue"),
                    labels = c("Fair", "Unfair")) + 
  guides(colour=guide_legend(title=NULL)) + 
  guides(fill=guide_legend(title=NULL)) + 
  scale_y_continuous(limits=c(1, 4.5), breaks=seq.int(1, 4.5, 0.5)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.01), legend.justification=c(0.99, 0.01), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig2


### Waiting time: fairDRL, w2DRL, Greedy, RHO 
names(fairDRL)
id <- c(1:nrow(fairDRL))
fair <- fairDRL$mean_wait
rho <- RHO$mean_wait
greedy <- Greedy$mean_wait
opt <- optDRL$mean_wait
w2gt <- w2DRL$mean_wait
w4gt <- w4DRL$mean_wait

X <- data.frame(id, fair, rho, greedy, opt, w2gt, w4gt)
names(X) <- c("id", "fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")

data <- data.frame(melt(X, id="id"))
case <- vector(mode="character", length=6*nrow(fairDRL))
case[1:(3*nrow(fairDRL))] <- 'Fair'
case[(3*nrow(fairDRL)+1):(6*nrow(fairDRL))] <- 'Unfair'

data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL"))
data$case <- factor(case, order=TRUE, 
                    levels=c('Fair', 'Unfair'))
data
summary(data)


fig3 <- ggplot(data, aes(x=variable, y=value, colour=case, fill=case)) + 
  stat_boxplot(geom="errorbar", width=0.25, size=0.6) + 
  geom_boxplot(alpha = 0.6, width=0.3, size=0.6, notch=FALSE, outlier.colour="red", 
               outlier.size=2.0) + 
  stat_summary(fun="mean", geom="point", shape=7, size=2.5, colour = "Black") +
  labs(x=NULL, y='waiting time (minute)', 
       title='(c) Distributions of average waiting time',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("OrangeRed", "DodgerBlue"),
                     labels = c("Fair", "Unfair")) + 
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue"),
                    labels = c("Fair", "Unfair")) + 
  guides(colour=guide_legend(title=NULL)) + 
  guides(fill=guide_legend(title=NULL)) + 
  scale_y_continuous(limits=c(5.5, 7), breaks=seq.int(5, 7, 0.25)) +
  theme_bw() +  
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.01), legend.justification=c(0.99, 0.01), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig3


### Detour: fairDRL, w2DRL, Greedy, RHO 
names(fairDRL)

id <- c(1:nrow(fairDRL))
fair <- fairDRL$mean_detour
rho <- RHO$mean_detour
greedy <- Greedy$mean_detour
opt <- optDRL$mean_detour
w2gt <- w2DRL$mean_detour
w4gt <- w4DRL$mean_detour

X <- data.frame(id, fair, rho, greedy, opt, w2gt, w4gt)
names(X) <- c("id", "fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL")

data <- data.frame(melt(X, id="id"))
case <- vector(mode="character", length=6*nrow(fairDRL))
case[1:(3*nrow(fairDRL))] <- 'Fair'
case[(3*nrow(fairDRL)+1):(6*nrow(fairDRL))] <- 'Unfair'

data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("fairDRL", "RHO", "Greedy", "optDRL", "w2DRL", "w4DRL"))
data$case <- factor(case, order=TRUE, 
                    levels=c('Fair', 'Unfair'))
data
summary(data)


fig4 <- ggplot(data, aes(x=variable, y=value, colour=case, fill=case)) + 
  stat_boxplot(geom="errorbar", width=0.25, size=0.6) + 
  geom_boxplot(alpha = 0.6, width=0.3, size=0.6, notch=FALSE, outlier.colour="red", 
               outlier.size=2.0) + 
  stat_summary(fun="mean", geom="point", shape=7, size=2.5, colour = "Black") +
  labs(x=NULL, y='detour distance (km)', 
       title='(d) Distributions of average detour distance',
       fill='Policy', color='Policy') + 
  scale_color_manual(values = c("OrangeRed", "DodgerBlue"),
                     labels = c("Fair", "Unfair")) + 
  scale_fill_manual(values = c("OrangeRed", "DodgerBlue"),
                    labels = c("Fair", "Unfair")) + 
  guides(colour=guide_legend(title=NULL)) + 
  guides(fill=guide_legend(title=NULL)) + 
  scale_y_continuous(limits=c(0.5, 2.5), breaks=seq.int(0.5, 2.5, 0.25)) +
  theme_bw() +  
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.99, 0.01), legend.justification=c(0.99, 0.01), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 
fig4


##export: Figure-test, 8 x 10
library(grid)
grid.newpage()
pushViewport(viewport(layout = grid.layout(2,2)))
vplayout <- function(x,y) {
  viewport(layout.pos.row = x, layout.pos.col = y)
}
print(fig1, vp = vplayout(1,1))
print(fig2, vp = vplayout(1,2))
print(fig3, vp = vplayout(2,1))
print(fig4, vp = vplayout(2,2))



###################
### rejection: fairDRL, w2DRL, Greedy, RHO 
names(fairDRL)
id <- c(1:nrow(fairDRL))
opt <- optDRL$rejection
w4gt <- w4DRL$rejection
w2gt <- w2DRL$rejection
greedy <- Greedy$rejection
fair <- fairDRL$rejection
rho <- RHO$rejection

X <- data.frame(id, opt, w4gt, w2gt, greedy, fair, rho)
names(X) <- c("id", "optDRL", "w4DRL", "w2DRL", "fairDRL", "RHO", "Greedy")

data <- data.frame(melt(X, id="id"))
data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("optDRL", "w4DRL", "w2DRL", "fairDRL", "RHO", "Greedy"))
summary(data)


ggplot(data, aes(x=variable, y=value)) + 
  stat_boxplot(geom="errorbar", width=0.25, size=0.6, colour = "DodgerBlue") + 
  geom_boxplot(colour = "DodgerBlue", fill = "DodgerBlue", alpha = 0.6, 
               width=0.3, size=0.6, notch=FALSE, outlier.colour="red", 
               outlier.size=2.0) + 
  stat_summary(fun="mean", geom="point", shape=7, size=2.5, fill="blue") +
  labs(x=NULL, y='rejection') + 
  theme_bw() +  
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.01, 0.99), legend.justification=c(0.01, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 







##################################################################
###################### Request services ########################
optDRL <- data.frame(read.csv("Request_records_optDRL.csv"))
str(optDRL)

w4DRL <- data.frame(read.csv("Request_records_w4DRL.csv"))
str(w4DRL)

w2DRL <- data.frame(read.csv("Request_records_w2DRL.csv"))
str(w2DRL)

fairDRL <- data.frame(read.csv("Request_records_fairDRL.csv"))
str(fairDRL)

Greedy <- data.frame(read.csv("Request_records_greedy.csv"))
str(Greedy)

RHO <- data.frame(read.csv("Request_records_RHO.csv"))
str(RHO)


names(fairDRL)
# [1] "id"       "time"     "cluster"  "area"    
# [5] "station"  "detour"   "wait"     "subsidy" 
# [9] "envy"     "scenario" "episode" 


#### Request subsidy ####
id <- c(1:nrow(fairDRL))
fair <- fairDRL$subsidy
rho <- RHO$subsidy

X <- data.frame(id, fair, rho)
names(X) <- c("id", "fairDRL", "RHO")

data <- data.frame(melt(X, id="id"))
data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("fairDRL", "RHO"))
summary(data)


ggplot(data, aes(x = value, y = ..density.., fill = variable)) +  
  geom_density(colour = "black", alpha = .5) + 
  scale_fill_manual(labels = c('centre demand', 'suburb demand'), 
                    values = c("OrangeRed", "DodgerBlue")) + 
  labs(x = 'subsidy', y = 'probability density', 
       title='(b) Distributions of subsidy', fill = "Policy") +
  scale_x_continuous(limits=c(0, 20), breaks = seq.int(0, 20, 2.5)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.01, 0.99), legend.justification=c(0.01, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 



#### Request envy ####
id <- c(1:nrow(fairDRL))
opt <- optDRL$envy
w4gt <- w4DRL$envy
w2gt <- w2DRL$envy
greedy <- Greedy$envy

X <- data.frame(id, opt, w4gt, w2gt, greedy, fair, rho)
names(X) <- c("id", "optDRL", "w4DRL", "w2DRL", "Greedy")

data <- data.frame(melt(X, id="id"))
data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("optDRL", "w4DRL", "w2DRL", "Greedy"))
summary(data)


ggplot(data, aes(x = value, y = ..density.., fill = variable)) +  
  geom_density(colour = "black", alpha = .5) + 
  scale_fill_manual(labels = c('centre demand', 'suburb demand'), 
                    values = c("OrangeRed", "DodgerBlue")) + 
  labs(x = 'envy', y = 'probability density', 
       title='(b) Distributions of envy', fill = "Policy") +
  scale_x_continuous(limits=c(0, 20), breaks = seq.int(0, 20, 2.5)) +
  theme_bw() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.01, 0.99), legend.justification=c(0.01, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 


#### Request wait ####
names(fairDRL)
id <- c(1:nrow(fairDRL))
opt <- optDRL$wait
w4gt <- w4DRL$wait
w2gt <- w2DRL$wait
greedy <- Greedy$wait
fair <- fairDRL$wait
rho <- RHO$wait

X <- data.frame(id, opt, w4gt, w2gt, greedy, fair, rho)
names(X) <- c("id", "optDRL", "w4DRL", "w2DRL", "fairDRL", "RHO", "Greedy")

data <- data.frame(melt(X, id="id"))
data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("optDRL", "w4DRL", "w2DRL", "fairDRL", "RHO", "Greedy"))
summary(data)


ggplot(data, aes(x=variable, y=value)) + 
  stat_boxplot(geom="errorbar", width=0.25, size=0.6, colour = "DodgerBlue") + 
  geom_boxplot(colour = "DodgerBlue", fill = "DodgerBlue", alpha = 0.6, 
               width=0.3, size=0.6, notch=FALSE, outlier.colour="red", 
               outlier.size=2.0) + 
  stat_summary(fun="mean", geom="point", shape=7, size=2.5, fill="blue") +
  labs(x = 'wait', y = 'probability density', 
       title='(b) Distributions of wait', fill = "Policy") +
  scale_x_continuous(limits=c(0, 20), breaks = seq.int(0, 20, 2.5)) +
  theme_bw() +  
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.01, 0.99), legend.justification=c(0.01, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 


#### Request detour ####
names(fairDRL)
id <- c(1:nrow(fairDRL))
opt <- optDRL$detour
w4gt <- w4DRL$detour
w2gt <- w2DRL$detour
greedy <- Greedy$detour
fair <- fairDRL$detour
rho <- RHO$detour

X <- data.frame(id, opt, w4gt, w2gt, greedy, fair, rho)
names(X) <- c("id", "optDRL", "w4DRL", "w2DRL", "fairDRL", "RHO", "Greedy")

data <- data.frame(melt(X, id="id"))
data$variable <- factor(data$variable, order=TRUE, 
                        levels=c("optDRL", "w4DRL", "w2DRL", "fairDRL", "RHO", "Greedy"))
summary(data)


ggplot(data, aes(x=variable, y=value)) + 
  stat_boxplot(geom="errorbar", width=0.25, size=0.6, colour = "DodgerBlue") + 
  geom_boxplot(colour = "DodgerBlue", fill = "DodgerBlue", alpha = 0.6, 
               width=0.3, size=0.6, notch=FALSE, outlier.colour="red", 
               outlier.size=2.0) + 
  stat_summary(fun="mean", geom="point", shape=7, size=2.5, fill="blue") +
  labs(x = 'detour', y = 'probability density', 
       title='(b) Distributions of detour', fill = "Policy") +
  scale_x_continuous(limits=c(0, 20), breaks = seq.int(0, 20, 2.5)) +
  theme_bw() +  
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_line(colour="Black", linetype="dashed", size=0.25),
        legend.position=c(0.01, 0.99), legend.justification=c(0.01, 0.99), 
        legend.background=element_rect(fill = "gray91"),
        legend.key=element_blank(), legend.key.height=unit(0.5,"cm"),
        legend.title=element_text(size=rel(1.0)),
        legend.text=element_text(size=rel(1.0)),
        plot.title = element_text(hjust=0.5, size=rel(1.25)),
        axis.title.x = element_text(size=rel(1.05)),
        axis.title.y = element_text(size=rel(1.05)),
        axis.text.x = element_text(size=rel(1.05)),
        axis.text.y = element_text(size=rel(1.05))) 

