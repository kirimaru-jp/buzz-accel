library(ggplot2); library(cowplot); library(latex2exp)


# Save a figure of Viterbi decoding state of dives to disk
# Inputs:
#  - dat0: dataframe of dive data and correspoding states
#  - n: number of states
#  - wght: the weights, i.e. the ratio of each decoded state by Viterbi algorithm
#  - tit: title of figure
#  - name: name of saved figure
drawFig = function(dat0, N, wght, tit="", name) 
{
    baseSize = 10
    dat = dat0[(MaxDepth <= 750) & (Duration <= 20)]
    subt = paste(rep("State",N),1:N,rep(':',N),
                             signif(100*wght, digits = 4),sep=" ")
    g1 = ggplot(data = dat,aes(x=Dive.Num, y=MaxDepth, colour=State, shape=State, size=State)) +
            geom_point(aes(size=as.factor(State))) +
            scale_size_manual(name = "States (%)", labels = subt,
                                                values = c(0.2,0.1,0.2,0.1)) +
            scale_colour_manual(name = "States (%)", labels = subt,
                                                values = c("#D55E00", "#009E73", "#0072B2","brown")) +
            scale_shape_manual(name = "States (%)", labels = subt,
                                             values = c(1,0,2,3)) + theme_classic(base_size=baseSize) +
            scale_y_continuous(expand = c(0.02,0.02),limits = c(0,790)) +
            # scale_y_reverse(expand = c(0.02,0.02),limits = c(790,0)) +
            # scale_x_continuous(expand = c(0.01,0.01),position = "top") +
            scale_x_continuous(expand = c(0.01,0.01)) +
            xlab("   ") + ylab("Depth (m)") # +
            # theme(plot.margin = unit(c(0,0,0,5), "mm"))
    g2 = ggplot(data = dat,aes(x=Dive.Num, y=Duration, colour=State, shape=State)) +
            geom_point(aes(size=as.factor(State))) +
            scale_size_manual(name = "States (%)", labels = subt,
                            	values = c(0.2,0.1,0.2,0.1)) +
            scale_colour_manual(name = "States (%)", labels = subt,
                                values = c("#D55E00", "#009E73", "#0072B2","brown")) +
            scale_shape_manual(name = "States (%)", labels = subt,
                                values = c(1,0,2,3)) + theme_classic(base_size=baseSize) +
            scale_y_continuous(expand = c(0.02,0.02),limits = c(0,22)) +
            scale_x_continuous(expand = c(0.01,0.01)) +
            xlab("   ") + ylab("Dive Dur.(min)")
    g3 = ggplot(data = dat,aes(x=Dive.Num, y=PostDiveDur, colour=State, shape=State)) +
            geom_point(aes(size=as.factor(State))) +
            scale_size_manual(name = "States (%)", labels = subt,
                                                values = c(0.2,0.1,0.2,0.1)) +
            scale_colour_manual(name = "States (%)", labels = subt,
                                                values = c("#D55E00", "#009E73", "#0072B2","brown")) +
            scale_shape_manual(name = "States (%)", labels = subt,
                                             values = c(1,0,2,3)) +
            theme_classic(base_size=baseSize) +
            theme(legend.position=c(0.8,1.12),
					axis.line = element_line(colour='black', size = 0.5),
					# axis.text.x=element_text(colour="black"),
					# axis.text.y=element_text(colour="black"),
					legend.key = element_rect(size = 0.5, color = NA),
					legend.text=element_text(size=8),
					legend.key.size = unit(0.5, 'lines'),
					legend.margin=margin(t=1, r=1, b=1, l=1, "mm"),
					plot.margin = unit(c(5,0,0,0), "mm"),
					legend.background = element_rect(size=0.5, linetype="solid", colour='gray')) +
            scale_y_continuous(expand = c(0.02,0.02),limits = c(0,150)) +
            scale_x_continuous(expand = c(0.01,0.01)) +
            xlab("Dive Number") + ylab("Post-Dive Dur.(min)")
    # https://stackoverflow.com/a/37345642
    pgrid  = plot_grid(g1+theme(legend.position = "none"),
                                            g2+theme(legend.position = "none"),
                                            g3, ncol = 1, align = 'v')
    save_plot(paste0("figures/", name, ".png"), pgrid,
				base_height = 1.8, ncol = 1, nrow = 3,
				base_aspect_ratio = 4,
				dpi = 600)
    save_plot(paste0("figures/", name, ".eps"), pgrid,
				base_height = 1.5, ncol = 1, nrow = 3,
				base_aspect_ratio = 4)
}


# Draw dives in different colors associating to states during 12 hours (Figure 8)
# dat: dataframe of depth data and coressponding states
drawDives.color = function(dat) {
    gp = ggplot(data = dat,aes(x=ID, y=Depth)) +
            geom_path(aes(color=as.factor(State)),size=0.8,group=1) +
            scale_color_manual(name = "", breaks=c('1', '2','3'),
                                                    values = c('0'='black','1'="#D55E00",
                                                                '2'="#009E73",'3'="#0072B2")) +
            theme_classic(base_size=25) +
            theme(legend.position=c(0.8,1.22),legend.title=element_blank(),
                        legend.text=element_text(size=25),
                        # legend.position="top", legend.box="horizontal",
                        axis.line = element_line(colour='black', size=0.5),
                        legend.margin=margin(t=0, r=1, b=0, l=0.3, "mm"),
                        plot.margin = unit(c(0,0,0,0), "mm"),
                        axis.text.x=element_text(colour='black'),
                        axis.text.y=element_text(colour='black')) +
            guides(color = guide_legend(nrow = 1,label.hjust=-0.05,title.position="top")) +
            scale_x_continuous(expand = c(0.03,0.03),position = "top") +
            scale_y_reverse(limits = c(max(dat$Depth)+20,0),expand = c(0.05,0.05) ) +
            xlab("Time (hours)") + ylab("Depth (m)")
    ggsave("Dives_rgb.png", gp, path="figures/",
                    width = 30, height = 10, units = c("cm"),
                    dpi = 600)
    ggsave("Dives_rgb.eps", gp, path="figures/",
                    width = 30, height = 10, units = c("cm"))
    gp
}


# Save figures of response variables and covariate processes (Figure 3), 
# based on the decoded hidden states from a model fitted to a dependent log-normal distribution (Model 1).
# Inputs:
#  - dat: dataframe of dive data and correspoding states
#  - n: number of states
#  - wght: the weights, i.e. the ratio of each decoded state by Viterbi algorithm
#  - tit: title of figure
drawState = function(dat, N, wght, tit) {
    # dat = copy(dat0)
    font.size = 14; base.ratio = 3.5; disc = 0.8; symb.size = 2.5
    # colour=State, shape=State, size=State

    g1 = ggplot(data = dat,aes(x=Dive.Num, y=MaxDepth)) +
            geom_point(aes(shape=as.factor(State),size=as.factor(State),color=as.factor(State) )) + geom_step(color='black',size=0.5)+
            scale_color_manual(name = "States", breaks=c('1', '2','3'),
                                values = c('0'='black','1'="#D55E00",'2'="#009E73",'3'="#0072B2")) +
            scale_size_manual(name = "States", values = rep(symb.size,3)) +
            scale_shape_manual(name = "States", values = c(16,15,17)) +
            theme_classic(base_size=font.size) +
            theme(legend.position='none',axis.line = element_line(colour = 'black', size = 0.5),
                        axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black"),
                        legend.key.size = unit(disc, 'lines')) +
            scale_y_continuous(expand = c(0.05,1.12)) + scale_x_continuous(expand = c(0.01,0.01)) +
            # scale_y_reverse(expand = c(0.05,1.12)) +
            # scale_x_continuous(expand = c(0.01,0.01),position = "top") +
            xlab("Dive Number") + ylab( bquote(atop("Maximum depth", X["1,t"]~" (m)")) )
    
		g2 = ggplot(data = dat,aes(x=Dive.Num, y=Duration)) +
            geom_point(aes(shape=as.factor(State),size=as.factor(State),color=as.factor(State))) + geom_step(color='black',size=0.5) +
                            scale_color_manual(name = "States", breaks=c('1', '2','3'),
                                                                                            values = c('0'='black','1'="#D55E00",
                                                                                                                                    '2'="#009E73",'3'="#0072B2")) +
            scale_size_manual(name = "States", values = rep(symb.size,3)) +
            scale_shape_manual(name = "States", values = c(16,15,17)) +
            theme_classic(base_size=font.size) +
            theme(legend.position=c(0.95,0.98),axis.line = element_line(colour = 'black', size = 0.5),
                        axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black"),
                        legend.key.size = unit(disc, 'lines')) +
            scale_y_continuous(expand = c(0.05,1.12)) + scale_x_continuous(expand = c(0.01,0.01)) +
            xlab("Dive Number") + ylab( bquote(atop("Duration of dive", X["2,t"]~" (minutes)")) )
    g3 = ggplot(data = dat,aes(x=Dive.Num, y=PostDiveDur)) +
            geom_point(aes(shape=as.factor(State),size=as.factor(State),color=as.factor(State))) + geom_step(color='black',size=0.5) +
            scale_color_manual(name = "States", breaks=c('1', '2','3'),
																values = c('0'='black','1'="#D55E00",
																						'2'="#009E73",'3'="#0072B2")) +
            scale_size_manual(name = "States", values = rep(symb.size,3)) +
            scale_shape_manual(name = "States", values = c(16,15,17)) +
            theme_classic(base_size=font.size) +
            theme(legend.position=c(-0.97,.75),axis.line = element_line(colour = 'black', size = 0.5),
                        axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black"),
                        legend.key.size = unit(disc, 'lines')) +
            scale_y_continuous(expand = c(0.05,1.12)) + scale_x_continuous(expand = c(0.01,0.01)) +
            xlab("Dive Number") + ylab( bquote(atop("Post-dive duration", X["3,t"]~" (minutes)")) )

    g4 = ggplot(data = dat,aes(x=Dive.Num, y=NoDDiveDur)) +
            geom_point(aes(shape=as.factor(State),size=as.factor(State),color=as.factor(State))) +
            geom_step(color='black',size=0.5) +
            # geom_line(color='black',size=0.5) +
            scale_color_manual(name = "States", breaks=c('1', '2','3'),
															values = c('0'='black','1'="#D55E00",
																					'2'="#009E73",'3'="#0072B2")) +
            scale_size_manual(name = "States", values = rep(symb.size,3)) +
            # scale_colour_manual(name = "States", values = c("#D55E00", "#009E73", "#0072B2")) +
            scale_shape_manual(name = "States", values = c(16,15,17)) +
            theme_classic(base_size=font.size) +
            theme(legend.position=c(-0.5,.75),axis.line = element_line(colour = 'black', size = 0.5),
                        axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black")) +
            scale_y_continuous(expand = c(0.05,1.12)) + scale_x_continuous(expand = c(0.01,0.01)) +
            xlab("Dive Number") + ylab( bquote(atop("Time since last", "deep dive "~tau[t]~" (minutes)")) )
    g5 = ggplot(data = dat,aes(x=Dive.Num, y=DDive)) +
            geom_point(aes(shape=as.factor(State),size=as.factor(State),color=as.factor(State))) + geom_step(color='black',size=0.5) +
            scale_color_manual(name = "States", breaks=c('1', '2','3'),
																values = c('0'='black','1'="#D55E00",
																						'2'="#009E73",'3'="#0072B2")) +        
            scale_size_manual(name = "States", values = rep(symb.size,3)) +
            scale_shape_manual(name = "States", values = c(16,15,17)) +
            theme_classic(base_size=font.size) +
            theme(legend.position=c(-0.97,.75),axis.line = element_line(colour = 'black', size = 0.5),
                        axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black"),
                        legend.key.size = unit(disc, 'lines')) +
            scale_y_continuous(expand = c(0.05,1.12)) + scale_x_continuous(expand = c(0.01,0.01)) +
            xlab("Dive Number") + ylab( bquote(atop("Number of deep", 'dives in a row'~d[t]))  )
    g6 = ggplot(data = dat,aes(x=Dive.Num, y=Hour)) +
            geom_point(aes(shape=as.factor(State),size=as.factor(State),color=as.factor(State))) + geom_step(color='black',size=0.5) +
            scale_color_manual(name = "States", breaks=c('1', '2','3'),
                               values = c('0'='black','1'="#D55E00",
																					'2'="#009E73",'3'="#0072B2")) +
            scale_size_manual(name = "States", values = rep(symb.size,3)) +
            scale_shape_manual(name = "States", values = c(16,15,17)) +
            theme_classic(base_size=font.size) +
            theme(legend.position=c(-0.97,.75),axis.line = element_line(colour = 'black', size = 0.5),
				axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black"),
				legend.key.size = unit(disc, 'lines')) +
            scale_y_continuous(expand = c(0.05,1.12)) + ylim(min(dat$Hour),ifelse(24-max(dat$Hour) < 2, 24,max(dat$Hour) )) +
            scale_x_continuous(expand = c(0.01,0.01)) +
            xlab("Dive Number") + ylab( bquote(atop("Time of day at initiation", 'of dive'~h[t]~" (hours)"))  )

    ppgrid = plot_grid(g1,g2,g3,g4,g5,g6, ncol = 1, align='v')
    save_plot(paste("figures/fig3.png"), ppgrid,
                            base_height = 2, ncol = 1, nrow = 6,
                            base_aspect_ratio = base.ratio,
                            dpi = 600)
    save_plot(paste("figures/fig3.eps"), ppgrid,
                            base_height = 2, ncol = 1, nrow = 6,
                            base_aspect_ratio = base.ratio)
    title = ggdraw() + draw_label(tit, fontface='bold', size=22)
    plot_grid(title, ppgrid, ncol=1, rel_heights=c(0.1, 1))
}




# Print the number with d decimal places after the point 
scaleFUN3 = function(x,d=3) sprintf( paste0("%.",d,"f"), x)


# Save a figure of all the distribution together on the same histograms (Figure 5 on the paper)
# including 4 distribution: Independent/Dependent Log-normal & Gamma
# Inputs:
# - dat: dataframe with maximum depth, dive duration, post-dive duration of all the dives
# - DL: fitted parameters of states of dependent log-normal distribution
# - DG: fitted parameters of states of dependent gamma distribution
# - IL: fitted parameters of states of independent log-normal distribution
# - IG: fitted parameters of states of independent gamma distribution
# - name: name of saved file
drawHisto_LN_GM = function(dat, DL,DG,IL,IG, name) {
  
  font = 25    

  # Mixed histogram for maximum depth
  gm_S = list(x=dat$MaxDepth, 
              muDL=DL$muMD,sigmaDL=DL$sigmaMD,deltaDL=DL$delta,
              muDG=DG$muMD,sigmaDG=DG$sigmaMD,deltaDG=DG$delta,
              muIL=IL$muMD,sigmaIL=IL$sigmaMD,deltaIL=IL$delta,
              muIG=IG$muMD,sigmaIG=IG$sigmaMD,deltaIG=IG$delta  )
  g1 = gg.mix_LN_GM(gm_S,'Maximum Depth','Depth (m)', LIM_Y=0.015,
                    no.legend=T,offset=19.5,prec=2)
  
  # Mixed histogram for dive duration
  gm_S = list(x=dat$Duration, 
              muDL=DL$muDT,sigmaDL=DL$sigmaDT,deltaDL=DL$delta,
              muDG=DG$muDT,sigmaDG=DG$sigmaDT,deltaDG=DG$delta,
              muIL=IL$muDT,sigmaIL=IL$sigmaDT,deltaIL=IL$delta,
              muIG=IG$muDT,sigmaIG=IG$sigmaDT,deltaIG=IG$delta  )
  g2  = gg.mix_LN_GM(gm_S,'Dive Duration','Duration (minutes)',
                      no.legend=T,hour=F)
  
  # Mixed histogram for post-dive duration
  gm_S = list(x=dat$PostDive, 
              muDL=DL$muPD,sigmaDL=DL$sigmaPD,deltaDL=DL$delta,
              muDG=DG$muPD,sigmaDG=DG$sigmaPD,deltaDG=DG$delta,
              muIL=IL$muPD,sigmaIL=IL$sigmaPD,deltaIL=IL$delta,
              muIG=IG$muPD,sigmaIG=IG$sigmaPD,deltaIG=IG$delta  )
  g3  = gg.mix_LN_GM(gm_S,'Post-dive Duration','Duration (minutes)', 
                      no.legend=F,LIM_X=20)
  

  ppgrid = plot_grid(plot_grid(g1, g2, g3, ncol = 2, nrow = 2),
                      ncol = 2, rel_widths = c(1,0.03)) +
                      draw_label("", x = 0.28, y = 1,
                      # draw_label("Correlated Log-normal dist.", x = 0.28, y = 1,
                      vjust = 1, hjust = 1, size = font, fontface = 'bold')

  save_plot(paste0("figures/", name, ".png"), ppgrid,
                      base_height = 6, ncol = 2, nrow = 2,
                      base_width  = 10, dpi = 600)
  save_plot(paste0("figures/", name, ".tiff"), ppgrid,
                      base_height = 4, ncol = 2,  nrow = 2, compression = "lzw",
                      base_width  = 7.75, dpi = 300)
  
  ppgrid
}


# Plot histogram of a response variable with distribution parameters
# Inputs:
# - gm_S: dataframe with data of response variable and fitted distribution parameters of 4 distributions Independent/Dependent Log-normal & Gamma
# - title: title of figure
# - xis: x-axis title
# - no.legend: true if allow legend, false otherwise
# - sum_curve: sum of all distribution curve or not?
# - hour: used for dive duration, whether the unit is hour
# - offset: offset of plotting, in case of maximum depth
# - prec: used for x-axis plotting, to arrange breaks and limits of x-axis
# - nbin: number of bins of histogram
# - LIM_X,LIM_Y: lower limits of x-axis and y-axis
# - state: if we do not plot all the states, but just some of them
# - font: font size of text in histogram
gg.mix_LN_GM = function(gm_S,title,xis='',no.legend=F,sum_curve=T,hour=F,offset=0,
                        prec=1,nbin=100,LIM_X=0,LIM_Y=0,state=0,font=35) {
    
  thick = 0.8
  x = with(gm_S,seq(min(x),max(x),len=1000))
  
  # data length = #State
  pars = with(gm_S,data.frame(State=as.character(seq(length(muDL))),
                              muDL,sigmaDL,deltaDL,
                              muDG,sigmaDG,deltaDG,
                              muIL,sigmaIL,deltaIL,
                              muIG,sigmaIG,deltaIG     ) )
  if (state!=0)
      pars = with(gm_S,data.frame(State=as.character(state),
                                  muDL,sigmaDL,deltaDL,
                                  muDG,sigmaDG,deltaDG,
                                  muIL,sigmaIL,deltaIL,
                                  muIG,sigmaIG,deltaIG ) )

  curve_state = data.table(x=rep(x,each=nrow(pars)),pars) # Make data

  
  if (hour) {
      curve_state$y   = with(curve_state,deltaDL*dlnorm((x-offset)*3600,muDL,sigmaDL)*3600)             # DL
      curve_state$yDG = with(curve_state,deltaDG*dgamma((x-offset)*3600,shape=muDG,scale=sigmaDG)*3600) # DG
      curve_state$yIL = with(curve_state,deltaIL*dlnorm((x-offset)*3600,muIL,sigmaIL)*3600)             # IL
      curve_state$yIG = with(curve_state,deltaIG*dgamma((x-offset)*3600,shape=muIG,scale=sigmaIG)*3600) # IG
  }
  else {
      curve_state$y   = with(curve_state,deltaDL*dlnorm((x-offset),muDL,sigmaDL))             # DL
      curve_state$yDG = with(curve_state,deltaDG*dgamma((x-offset),shape=muDG,scale=sigmaDG)) # DG
      curve_state$yIL = with(curve_state,deltaIL*dlnorm((x-offset),muIL,sigmaIL))             # IL
      curve_state$yIG = with(curve_state,deltaIG*dgamma((x-offset),shape=muIG,scale=sigmaIG)) # IG
  }
  
  if (nrow(pars) > 1) { # http://stackoverflow.com/a/28267510
      ind = c(rep(nrow(pars), floor(NROW(curve_state)/nrow(pars))),
                      floor(NROW(curve_state)%%nrow(pars)) )
      ind = rep(1:length(ind), times = ind)
      curve_sum = data.table(x=x,
                             y  =as.vector(apply( data.frame(curve_state$y),  2,function(z) tapply(z,ind,sum) )),
                             yDG=as.vector(apply( data.frame(curve_state$yDG),2,function(z) tapply(z,ind,sum) )),
                             yIL=as.vector(apply( data.frame(curve_state$yIL),2,function(z) tapply(z,ind,sum) )),
                             yIG=as.vector(apply( data.frame(curve_state$yIG),2,function(z) tapply(z,ind,sum) )) )
  }
  
  dt = melt(curve_state[,c('x','State','y', 'yDG', 'yIL', 'yIG' ) ],
                      measure.vars = c('y', 'yDG', 'yIL', 'yIG'),
                      variable.name = "Model", value.name = "y" )
  
  gp = ggplot(data.frame(x=gm_S$x),aes(x,y=..density..)) +
              geom_histogram(fill=NA,color="grey",bins=nbin,boundary=0)+
      geom_line(data=dt,aes(x,y,color=State,linetype=Model),alpha=1,size=thick) + scale_y_continuous( labels=scaleFUN3 ) +
      scale_colour_manual(values = c("1"="#D55E00", "2"="#009E73",
                                     "3"="#0072B2", "4"="brown")) +
              scale_linetype_manual(labels = c("DL","DG",'IL','IG' ),
                                    values = c("y"="solid", "yDG"="dashed",
                                                "yIL"="dotted", "yIG"="dotdash")) +
              xlab(xis) + ylab("Density") + ggtitle(title)
  if (LIM_X > 0)
      gp = gp + scale_x_continuous(limits=c(0,LIM_X))
  else {
      if (hour) gp = gp + scale_x_continuous(limits=c(0,0.3)) +
                                              scale_y_continuous(limits=c(0,11))
      else gp = gp + scale_x_continuous( limits=c(0,round(max(x)-(10^prec)+5,-prec)),
                                           breaks=seq(0,round(max(x)-(10^prec)+5,-prec),
                                           by=round(max(x)-(10^prec)+5,-prec)/4) )
  }
  if (LIM_Y > 0)
      gp = gp + scale_y_continuous(limits=c(0,LIM_Y))
  
  gp = gp + theme_classic(base_size=font) + # stackoverflow.com/a/39112908
                      theme(legend.position=c(1.5,0.5),
							legend.key.width=unit(2.3,"cm"),
							legend.text=element_text(size=40),
							legend.margin=unit(0, "cm"),
							axis.line   = element_line(colour='black',size=thick),
							axis.text.x = element_text(colour='black',size=font),
							axis.text.y=element_blank(),
							axis.ticks.y=element_blank()
							) +
							guides( guide_legend(nrow=4),
								override.aes = list(shape=3.5) )
  
  
  cs = melt(curve_sum,
              measure.vars = c('y', 'yDG', 'yIL', 'yIG'),
              variable.name = "Model", value.name = "y" )
  if (nrow(pars) > 1 && sum_curve)
      gp = gp + geom_line(data=cs,aes(x,y,linetype=Model),
                                              color='black', #color='#CC79A7',
                                              alpha=1,size=thick)    
  if (no.legend)
      return ( gp+theme(legend.position = "none") )
      
  gp
}

