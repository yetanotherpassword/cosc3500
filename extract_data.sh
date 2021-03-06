rm mfile.m
echo "clear all" >> mfile.m
echo "close all" >> mfile.m
procs="add_mat add_scalar mult_scalar piecewisemult set_diff2_piecewisemult3 set_matmult set_mult1_add2_mat set_mult1_add2_scalars set_transpose"
declare -A  smax_d=()
declare -A  sall_d=()
declare -A  savg_d=()
declare -A  smax_r=()
declare -A  sall_r=()
declare -A  savg_r=()
declare -a  plot_options
plot_options[1]="\"o:\""
plot_options[2]="\"*-.\""
plot_options[3]="\"s-\""
plot_options[4]="\"v--\""
plot_options[5]="\"d:\""
plot_options[6]="\"+-.\""
cnt=1
names="{"
grep "\"784 " slurm-1*.out | awk -F\" '{print $2}' | tr " " "_" | sort -u | awk '{  dir=$1; print "if [ -d \x22"dir"\x22 ]; then"; print "  rm -rf "dir; print "fi"; print "mkdir "dir }' > doit
grep "\"784 " slurm-1* | sed "s/ /_/g" | awk -F: '{a=match($2,"\""); dir = substr($2,a+1,length($2)-a-2);  print "cp "$1" "dir}' >> doit
remove=`grep "CANCELLED\|Aborted\|Killed\|Valgrind cannot continue" slurm-1* | awk -F: '{print $1}'`
running=`squeue -u s4604901| tail -n +2 | awk '{print "slurm-"$1".out"}'`
removeall=`echo -e "$remove\n$running" | uniq | tr " " "\n"`
echo "removeall="$removeall
for p in $removeall
do
    sed -i "/$p/d" doit
    echo "Removing $p"
done
chmod +x doit
./doit
laylist=""
dirs=`grep "\"784 " slurm-1* | sed "s/ /_/g" | awk -F: '{a=match($2,"\""); dir = substr($2,a+1,length($2)-a-2); print dir}'| sort -u`
for i in $dirs
do
  layr=`echo $i | awk -F_ '{print NF}'`
  files=`ls $i`
  if [[ -z "$files" ]]; then
     continue
  fi
  echo $i has $files
  echo
  for j in $files
  do
if [[ "$i/$j" == "784_500_300_10/slurm-1305.out" ]]; then
  set -x
  echo "$i/$j is ================================================="
fi
if [[ "$i/$j" == "784_500_300_10/slurm-1368.out" ]]; then
  set -x
  echo "$i/$j is ================================================="
fi
     if grep -q "Killed\|Valgrind cannot continue" $i/$j ; then
         rm -v $i/$j
     fi
     if [ -f $i/$j ]; then
            type=`grep "Build type is :" $i/$j | awk -F: '{print $2}'| tr -s " " | tr -d " "`
            if [ -z "$type" ]; then
                echo "Build type not found in $i/$j, try older ref"

                if grep  -i serial $i/$j; then
                   alttyp="Parallel"
                   type="Serial"
                elif  grep  -i parallel $i/$j; then
                   alttyp="Serial"
                   type="Parallel"
                else
                   echo "Alt Build type not found in $i/$j, skipping file"
                   continue
                fi
            elif echo $type | grep -q -i serial ; then
               alttyp="Parallel"
            elif echo $type | grep -q -i parallel ; then
               alttyp="Serial"
            else
               echo "Build type not found so skipping "
               continue
            fi
            epc=`grep "Epochs in Training" $i/$j | awk -F:  '{print $2}'| tr -s " " | tr -d " "`
            if [ -z "$epc" ]; then
               epc=`grep "Training epochs"  $i/$j | awk '{ print $NF }'`
               if [ -z "$epc" ]; then
                  continue
               fi
            fi
            matefil=`grep "Build type is : $alttyp" $i/*.out | awk -F: '{print $1}' | sort -u | tr '\n' ' '`
echo "i******************************** matefil=$matefil"
            if [[ -z "$matefil" ]]; then
               echo "Skipping as no matching $alttyp file found for $build for $i on Epoch $epc"
               continue
            elif ! grep "Training epochs set to $epc\|Epochs in Training : $epc" $matefil; then
               echo "Skipping as no matching $alttyp Epoch $epc exists in $type  $matefil"
               continue
            fi
            fil=$i/${type}_E${epc}_$j
            filout=${fil}.csv
            cp -v $i/$j $fil
            if [ -f "$filout" ]; then
               rm $filout
            fi
            tot_corr=`grep "Total Correct" $fil | uniq | awk -F: '{ print $2 }'`
            tot_tim=`grep "Total Time" $fil | uniq | awk -F: '{ print $2 }' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            tot_trng_tim=`grep "Total Train Time" $fil | uniq | awk -F: '{ print $2 }' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            tot_test_tim=`grep "Total Test Time" $fil | uniq | awk -F: '{ print $2 }' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            network=$i
            pf="L${layr}_E${epc}"
            if [ -z "$smax_d[$pf]" ]; then
                smax_d[$pf]="figure;\n"
            fi
            if [ -z "$sall_d[$pf]" ]; then
              sall_d[$pf]="figure;\n"
            fi
            if [ -z "$savg_d[$pf]" ]; then
              savg_d[$pf]="figure;\n"
            fi
            if [ -z "$smax_r[$pf]" ]; then
              smax_r[$pf]="figure;\n"
            fi
            if [ -z "$sall_r[$pf]" ]; then
              sall_r[$pf]="figure;\n"
            fi
            if [ -z "$savg_r[$pf]" ]; then
              savg_r[$pf]="figure;\n"
            fi
            prefix1="${pf}_${type}"
  if echo $laylist | grep -w $layr ; then
    echo "Seen this layer before"
  else
     echo -e "L${layr}_all_tots_list=[]; \n" >>  $filout
     echo -e "L${layr}_all_trng_list=[]; \n" >>  $filout
     echo -e "L${layr}_all_test_list=[]; \n" >>  $filout
     laylist="$laylist $layr"
  fi
            if [ -n "$initit1" ]; then
                echo -e $initit1 >> $filout
                echo -e $initit2 >> $filout
                echo -e $initit2 >> $filout
                initit1=""
                initit2=""
                initit3=""
            fi
            bldver=`grep "Build done" $fil`
            echo "${prefix1}_Build=\"$bldver\";" >> $filout
            echo "${prefix1}_Type=\"$type\";" >> $filout
            echo "${prefix1}_tot_corr=\"$tot_corr\";" >> $filout
            echo "${prefix1}_network=\"$network\";" >> $filout
            echo "${prefix1}_Layers=$layr;" >> $filout
            echo "${prefix1}_Epochs=$epc;" >> $filout
            if [[ "$type" == "Serial" ]]; then
               comp1="PtoS"
               comp2="PdiffS"
               prefix2="${pf}_${comp1}"
               prefix3="${pf}_${comp2}"
               altprefix1="${pf}_Parallel"
               echo -e "${prefix1}_tot_time=$tot_tim;\n ${pf}_tot_PtoS=${prefix1}_tot_time/${altprefix1}_tot_time; \n  ${pf}_tot_PdiffS=${prefix1}_tot_time-${altprefix1}_tot_time; \n ${pf}_all_tots=[   ${pf}_tot_PtoS ${pf}_tot_PdiffS];\n" >> $filout
               echo -e "L${layr}_all_tots_list=[L${layr}_all_tots_list; \n  ${pf}_all_tots $epc];" >> $filout
               echo -e "${prefix1}_tot_train=$tot_trng_tim; \n ${pf}_trg_PtoS=${prefix1}_tot_train/${altprefix1}_tot_train; \n  ${pf}_trg_PdiffS=${prefix1}_tot_train-${altprefix1}_tot_train; \n ${pf}_all_trgs=[   ${pf}_trg_PtoS ${pf}_trg_PdiffS];\n" >> $filout
               echo -e "L${layr}_all_trng_list=[L${layr}_all_trng_list; \n  ${pf}_all_trgs $epc];" >> $filout
               echo -e "${prefix1}_tot_test=$tot_test_tim; \n  ${pf}_tst_PtoS=${prefix1}_tot_test/${altprefix1}_tot_test; \n   ${pf}_tst_PdiffS=${prefix1}_tot_test-${altprefix1}_tot_test;\n   ${pf}_all_tsts=[   ${pf}_tst_PtoS ${pf}_tst_PdiffS];\n " >> $filout
               echo -e "L${layr}_all_test_list=[L${layr}_all_test_list; \n  ${pf}_all_tsts $epc];" >> $filout
            else
               echo "${prefix1}_tot_time=$tot_tim;" >> $filout
               echo "${prefix1}_tot_train=$tot_trng_tim;" >> $filout
               echo "${prefix1}_tot_test=$tot_test_tim;" >> $filout
            fi
            maxrname="max_ratios_${pf}"
            maxdname="max_diffs_${pf}"
            allrname="all_ratios_${pf}"
            alldname="all_diffs_${pf}"
            avgrname="avg_ratios_${pf}"
            avgdname="avg_diffs_${pf}"
            maxratios="$maxrname=["
            maxdiffs="$maxdname=["
            allratios="$allrname=["
            alldiffs="$alldname=["
            avgratios="$avgrname=["
            avgdiffs="$avgdname=["
            xtic="{"
     for k in $procs
     do
            xtic="$xtic \"$k\" "
            grep -A 6 -w $k $fil > $i/tmp
            maxt=`grep Max $i/tmp | awk -F: '{print $2}' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            allt=`grep All $i/tmp | awk -F: '{print $2}' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            avgt=`grep Avg $i/tmp | awk -F: '{print $2}' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            callsnum=`grep Calls $i/tmp | awk -F: '{print $2}'`
            pref="${prefix1}_${k}"
            pref2="${prefix2}_${k}"
            pref3="${prefix3}_${k}"
            altpref="${altprefix1}_${k}"
            if [[ "$type" == "Serial" ]]; then
                maxt2="${pref}_max=$maxt;\n  ${pref2}_max=$maxt/${altpref}_max ; \n     ${pref3}_max=$maxt-${altpref}_max ; \n ${pf}_all_max= [  ${pref2}_max ${pref3}_max ];\n  "
                allt2="${pref}_all=$allt; \n ${pref2}_all=$allt/${altpref}_all ;  \n    ${pref3}_all=$allt-${altpref}_all ;\n  ${pf}_all_all= [  ${pref2}_all ${pref3}_all ];\n  "
                avgt2="${pref}_avg=$avgt; \n ${pref2}_avg=$avgt/${altpref}_avg ;  \n    ${pref3}_avg=$avgt-${altpref}_avg ;\n  ${pf}_all_avg= [ ${pref2}_avg ${pref3}_avg ]; \n "
                maxratios="${maxratios}${pref2}_max;\n"
                maxdiffs="${maxdiffs}${pref3}_max;\n"
                allratios="${allratios}${pref2}_all;\n"
                alldiffs="${alldiffs}${pref3}_all;\n"
                avgratios="${avgratios}${pref2}_avg;\n"
                avgdiffs="${avgdiffs}${pref3}_avg;\n"
                names="${names}\"${pref}\"\n"
            else
                maxt2="${pref}_max=$maxt;\n "
                allt2="${pref}_all=$allt;\n "
                avgt2="${pref}_avg=$avgt;\n "
            fi
            echo -e $maxt2 >> $filout
            echo -e $allt2 >> $filout
            echo -e $avgt2 >> $filout
     done
     newpf=`echo $pf |  sed 's/_/ /g'`
     xtick="$xtic } "
     xtic=`echo $xtick | sed 's/_/\\\\_/g'`
     opt=${plot_options[$cnt]}
     net=`echo $i | sed 's/_/\\\\_/g'`
###################################
     tmpmax_d="$maxdiffs ];\n"
     rmnull=`echo $tmpmax_d | sed "/\[ \];/d"`
     if [ -n "$rmnull" ]; then
         smax_d[$pf]="$smax_d[$pf] $rmnull\n x=[1:1:size($maxdname,1)]; \n  plot(x,$maxdname*100-100,$opt); \n hold on \n title(\"Diffs of Max Time for Routines : Network Layers : ${layr} : $net\"); \nxlabel(\"Routine\")\n set(gca,'XTick',x) \n set(gca,'XTickLabel',$xtic) \n ylabel(\"Max Serial minus Max Parallel (Secs)\")\n "
     fi

     tmpall_d="$alldiffs ];\n"
     rmnull=`echo  $tmpall_d | sed "/\[ \];/d"`
     if [ -n "$rmnull" ]; then
         sall_d[$pf]="$sall_d[$pf] $rmnull\n  x=[1:1:size($alldname,1)]; \n plot(x,$alldname*100-100,$opt);\n  hold on \n title(\"Diffs of All Time for Routines : Network Layers : ${layr} : $net\"); \nxlabel(\"Routine\")\n set(gca,'XTick',x) \n set(gca,'XTickLabel',$xtic) \n ylabel(\"All Serial minus All Parallel (Secs)\")\n "
     fi

     tmpavg_d="$avgdiffs ];\n"
     rmnull=`echo  $avgmax_d | sed "/\[ \];/d"`
     if [ -n "$rmnull" ]; then
         savg_d[$pf]="$savg_d[$pf] $rmnull\n  x=[1:1:size($avgdname,1)]; \n plot(x,$avgdname*100-100,$opt);\n  hold on \n  title(\"Diffs of Avg Time for Routines : Network Layers : ${layr} : $net\"); \nxlabel(\"Routine\")\n set(gca,'XTick',x) \n set(gca,'XTickLabel',$xtic) \n ylabel(\"Avg Serial minus Avg Parallel (Secs)\")\n "
     fi
     tmpmax_r="$maxratios ];\n"
     rmnull=`echo  $tmpmax_r | sed "/\[ \];/d"`
     if [ -n "$rmnull" ]; then
         smax_r[$pf]="$smax_r[$pf] $rmnull\n  x=[1:1:size($maxrname,1)]; \n plot(x,$maxrname*100-100,$opt);\n  hold on \n  title(\"Parallel and Serial Max Time Comparisons per routine :  Network Layers : ${layr} : $net\"); \nxlabel(\"Routine\")\n set(gca,'XTick',x) \n set(gca,'XTickLabel',$xtic) \n ylabel(\"Max Routine Time Parallel faster than Serial (%)\")\n "
     fi
     tmpall_r="$allratios ];\n"
     rmnull=`echo  $tmpall_r | sed "/\[ \];/d"`
     if [ -n "$rmnull" ]; then
         sall_r[$pf]="$sall_r[$pf] $rmnull\n x=[1:1:size($allrname,1)]; \n plot(x,$allrname*100-100,$opt);\n hold on \n  title(\"Parallel and Serial All Time Comparisons per routine  :  Network Layers : ${layr} : $net\"); \nxlabel(\"Routine\")\n set(gca,'XTick',x) \n set(gca,'XTickLabel',$xtic) \n ylabel(\"All Serial:Parallel Time Ratios (%)\")\n "
     fi

     tmpavg_r="$avgratios ];\n"
     rmnull=`echo  $tmpavg_r | sed "/\[ \];/d"`
     if [ -n "$rmnull" ]; then
         savg_r[$pf]="$savg_r[$pf] $rmnull  x=[1:1:size($avgrname,1)]; \n plot(x,$avgrname*100-100,$opt);\n hold on \n  title(\"Parallel and Serial Avg Time Comparisons per routine  :  Network Layers : ${layr} : $net\"); \nxlabel(\"Routine\")\n set(gca,'XTick',x) \n set(gca,'XTickLabel',$xtic) \n ylabel(\"Avg Serial:Parallel Time Ratios (%)\")\n"
     fi
     let "cnt=cnt+1"
     if [ $cnt -eq 7 ]; then
        cnt=1
     fi
            real=`tail -20 $fil | grep real | awk '{print $2}'`
            user=`tail -20 $fil |grep user | awk '{print $2}'`
            sys=`tail -20 $fil |grep sys | awk '{print $2}'`
            if [ -n "$real" ]; then
               echo  "${prefix1}_real=\"$real\";" >>  $filout
               echo  "${prefix1}_sys=\"$sys\";" >>  $filout
               echo  "${prefix1}_user=\"$user\";" >>  $filout
            fi
     fi
set +x
  done
  newfiles=`ls  $i/*.csv |  cut -d"-" -f1 |sort -u`
  pfiles=""
  for z in $newfiles
  do
     b=`basename $z`
     latest=`ls "${i}" | grep $b | grep csv | cut -d"-" -f2 | sort | tail -1`
     thefile=`ls ${i}/${b}-${latest}`
     if echo $thefile | grep -q Parallel ; then
        pfiles="${pfiles}${thefile}\n"
     else
       com=`basename $thefile | awk -F"_" '{print substr($0,length($1)+1)}' | cut -d"-" -f1`
       matchp=`echo -e $pfiles | grep $com` 
       
#       if [ -n "$hdr" ]; then
#          echo -e $hdr >> $filout
#          hdr=""
#       fi
       if [[ -n "$matchp" && -n "$thefile" ]]; then
          cat $matchp  >> mfile.m
          cat $thefile >> mfile.m
       else
          echo "No match for $thefile, skipping"
       fi
     fi
  done
done

###################################################
neworder=`echo  "${!smax_r[@]}" | sed "s/[LE]//g" | tr ' ' '\n' | sort -t_ -k1,1n -k2,2n | sed "s/^/L/g; s/_/_E/g" | tr '\n' ' '`
starting=`echo $neworder | awk '{print $1}'`
startl=`echo $starting | cut -d"_" -f1`
starte=`echo $starting | cut -d"_" -f2 | awk '{print "\x22"$1"\x22,"}'`
for i in ${neworder}
do
  L=`echo $i|cut  -d"_" -f1`
  E=`echo $i|cut  -d"_" -f2`
  if [[ "$L" != "$startl" ]]; then
     startl=$L
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  
      starte="\"$E\","
  else
    if [[ $starte != \"${E}\", ]]; then
     starte="$starte \"$E\","
    fi
  fi
b=`echo ${smax_r[$i]} | sed "s/^\[[0-9LE_]*\]//g" | sed "s/^[0-9]* //g"`
  echo -e "$b" >> mfile.m
done
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  

########################
neworder=`echo  "${!smax_d[@]}" | sed "s/[LE]//g" | tr ' ' '\n' | sort -t_ -k1,1n -k2,2n | sed "s/^/L/g; s/_/_E/g" | tr '\n' ' '`
starting=`echo $neworder | awk '{print $1}'`
startl=`echo $starting | cut -d"_" -f1`
starte=`echo $starting | cut -d"_" -f2 | awk '{print "\x22"$1"\x22,"}'`
for i in ${neworder}
do
  L=`echo $i|cut  -d"_" -f1`
  E=`echo $i|cut  -d"_" -f2`
  if [[ "$L" != "$startl" ]]; then
     startl=$L
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  
      starte="\"$E\","
  else
    if [[ $starte != \"${E}\", ]]; then
     starte="$starte \"$E\","
    fi
  fi
b=`echo ${smax_d[$i]} | sed "s/^\[[0-9LE_]*\]//g" | sed "s/^[0-9]* //g"`
  echo -e "$b" >> mfile.m
done
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  



########################
neworder=`echo  "${!sall_r[@]}" | sed "s/[LE]//g" | tr ' ' '\n' | sort -t_ -k1,1n -k2,2n | sed "s/^/L/g; s/_/_E/g" | tr '\n' ' '`
starting=`echo $neworder | awk '{print $1}'`
startl=`echo $starting | cut -d"_" -f1`
starte=`echo $starting | cut -d"_" -f2 | awk '{print "\x22"$1"\x22,"}'`
for i in ${neworder}
do
  L=`echo $i|cut  -d"_" -f1`
  E=`echo $i|cut  -d"_" -f2`
  if [[ "$L" != "$startl" ]]; then
     startl=$L
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  
      starte="\"$E\","
  else
    if [[ $starte != \"${E}\", ]]; then
     starte="$starte \"$E\","
    fi
  fi
b=`echo ${sall_r[$i]} | sed "s/^\[[0-9LE_]*\]//g" | sed "s/^[0-9]* //g"`
  echo -e "$b" >> mfile.m
done
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  



########################
neworder=`echo  "${!sall_d[@]}" | sed "s/[LE]//g" | tr ' ' '\n' | sort -t_ -k1,1n -k2,2n | sed "s/^/L/g; s/_/_E/g" | tr '\n' ' '`
starting=`echo $neworder | awk '{print $1}'`
startl=`echo $starting | cut -d"_" -f1`
starte=`echo $starting | cut -d"_" -f2 | awk '{print "\x22"$1"\x22,"}'`
for i in ${neworder}
do
  L=`echo $i|cut  -d"_" -f1`
  E=`echo $i|cut  -d"_" -f2`
  if [[ "$L" != "$startl" ]]; then
     startl=$L
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  
      starte="\"$E\","
  else
    if [[ $starte != \"${E}\", ]]; then
     starte="$starte \"$E\","
    fi
  fi
b=`echo ${sall_d[$i]} | sed "s/^\[[0-9LE_]*\]//g" | sed "s/^[0-9]* //g"`
  echo -e "$b" >> mfile.m
done
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  



########################
neworder=`echo  "${!savg_r[@]}" | sed "s/[LE]//g" | tr ' ' '\n' | sort -t_ -k1,1n -k2,2n | sed "s/^/L/g; s/_/_E/g" | tr '\n' ' '`
starting=`echo $neworder | awk '{print $1}'`
startl=`echo $starting | cut -d"_" -f1`
starte=`echo $starting | cut -d"_" -f2 | awk '{print "\x22"$1"\x22,"}'`
for i in ${neworder}
do
  L=`echo $i|cut  -d"_" -f1`
  E=`echo $i|cut  -d"_" -f2`
  if [[ "$L" != "$startl" ]]; then
     startl=$L
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  
      starte="\"$E\","
  else
    if [[ $starte != \"${E}\", ]]; then
     starte="$starte \"$E\","
    fi
  fi
b=`echo ${savg_r[$i]} | sed "s/^\[[0-9LE_]*\]//g" | sed "s/^[0-9]* //g"`
  echo -e "$b" >> mfile.m
done
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  



########################
neworder=`echo  "${!savg_d[@]}" | sed "s/[LE]//g" | tr ' ' '\n' | sort -t_ -k1,1n -k2,2n | sed "s/^/L/g; s/_/_E/g" | tr '\n' ' '`
starting=`echo $neworder | awk '{print $1}'`
startl=`echo $starting | cut -d"_" -f1`
starte=`echo $starting | cut -d"_" -f2 | awk '{print "\x22"$1"\x22,"}'`
for i in ${neworder}
do
  L=`echo $i|cut  -d"_" -f1`
  E=`echo $i|cut  -d"_" -f2`
  if [[ "$L" != "$startl" ]]; then
     startl=$L
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg);  \n figure \n" >>  mfile.m  
      starte="\"$E\","
  else
    if [[ $starte != \"${E}\", ]]; then
     starte="$starte \"$E\","
    fi
  fi
b=`echo ${savg_d[$i]} | sed "s/^\[[0-9LE_]*\]//g" | sed "s/^[0-9]* //g"`
  echo -e "$b" >> mfile.m
done
     leg=`echo $starte | sed "s/,$//g"`
     echo -e "legend($leg); \n figure \n" >>  mfile.m  

echo -n "names=" >> mfile.m
echo -e $names >> mfile.m
echo "};" >> mfile.m
#grep "=\[\]" mfile.m > defs
#sed -i '/=\[\]/d'  mfile.m
#cat defs mfile.m  > mfile2.m

#set | grep $prefix1
