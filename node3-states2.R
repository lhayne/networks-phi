library(readr)
library(ggplot2)
setwd('/Users/michael/networks-phi')

d <- read_csv('nodes3-states2-delete.csv')
pval <- t.test(d$sleep,d$awake)
pvalue <- format(round(pval$p.value, 2), nsmall = 2)

# make the data tiddy
phi = c(d$sleep, d$awake)
state = c(rep('sleep',dim(d)[1]),rep('awake',dim(d)[1]))
dd <- data.frame(phi,state)

# violin plot
ggplot(data=dd,aes(y=phi,x=state,group=state)) + geom_violin(aes(y=phi,x=state,color=state)) + ggtitle(paste('3 Nodes p=',pvalue))+ 
  stat_summary(fun.y=median, geom="point", size=2, color="red")+theme_bw()+
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggsave('nodes3-violin.png')

# normal boring box plot
ggplot(data=dd,aes(y=phi,x=state,group=state)) + geom_boxplot(aes(y=phi,x=state,color=state)) + ggtitle(paste('3 Nodes p=',pvalue))+ theme_bw()+
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggsave('nodes3-box.png')
