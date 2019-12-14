library(readr)
library(ggplot2)
setwd('/Users/michael/networks-phi')

d <- read_csv('nodes3-all-states.csv')
pval <- t.test(d$sleep,d$awake)
pvalue <- format(round(pval$p.value, 2), nsmall = 2)
pvalue <- pval$p.value
# make the data tiddy
phi = c(d$sleep, d$awake)
starting_state = c(d$states, d$states)
state = c(rep('sleep',dim(d)[1]),rep('awake',dim(d)[1]))
dd <- data.frame(phi,state,starting_state)

pvs = c()
for(s in unique(d$states)){
  
  temp <- d[d$states == s,]
  pvs = append(pvs,t.test(temp$sleep,temp$awake)$p.value)
}
print(pvs)
pval_df <- data.frame(p_values=pvs,states=unique(d$states))

ggplot(data = pval_df, aes(y=p_values,x=states)) + geom_point() + theme_bw() + ggtitle('P-value of All Possible Starting States with 3 Nodes') + geom_hline(yintercept=0.05,color='red')
ggsave('node3_all_pvalues.png')

# violin plot
ggplot(data=dd,aes(y=phi,x=state,group=state)) + geom_violin(aes(y=phi,x=state,color=state)) + ggtitle(paste('3 Nodes p=',pvalue))+ 
  stat_summary(fun.y=median, geom="point", size=2, color="red")+
  theme_bw()+
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
ggsave('nodes3-violin-all-states.png')

# normal boring box plot
ggplot(data=dd,aes(y=phi,x=state,group=state)) + geom_boxplot(aes(y=phi,x=state,color=state)) + ggtitle(paste('3 Nodes p=',pvalue))+
  theme_bw()+
  theme(axis.title.x=element_blank(),
      axis.text.x=element_blank(),
      axis.ticks.x=element_blank())
ggsave('nodes3-box-all-states')
