rm mfile.m
procs="add_mat add_scalar mult_scalar piecewisemult set_diff2_piecewisemult3 set_matmult set_mult1_add2_mat set_mult1_add2_scalars set_transpose"
maxratios="["
maxdiffs="["
allratios="["
alldiffs="["
avgratios="["
avgdiffs="["
allratios="["
alldiffs="["
names="{"
grep "\"784 " slurm-1*.out | awk -F\" '{print $2}' | tr " " "_" | sort -u | awk '{  dir=$1; print "if [ -d \x22"dir"\x22 ]; then"; print "  rm -rf "dir; print "fi"; print "mkdir "dir }' > doit
grep "\"784 " slurm-1* | sed "s/ /_/g" | awk -F: '{a=match($2,"\""); dir = substr($2,a+1,length($2)-a-2);  print "cp "$1" "dir}' >> doit
remove=`grep "Aborted\|Killed\|Valgrind cannot continue" slurm-1* | awk -F: '{print $1}'`
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
dirs=`grep "\"784 " slurm-1* | sed "s/ /_/g" | awk -F: '{a=match($2,"\""); dir = substr($2,a+1,length($2)-a-2); print dir}'| sort -u`
for i in $dirs
do
  layr=`echo $i | awk -F_ '{print NF}'`
  files=`ls $i`
#  epclst=`grep "Epochs in Training" $i/$files | awk -F: '{print $3}'|sort -u -n`
#  hdr=""
#  for e in $epclst
#  do
#set -x
#     hdr="$hdr L${layr}_E${e}_all_tots=[];\nL${layr}_E${e}_all_trgs=[];\n L${layr}_E${e}_all_tsts=[];\n L${layr}_E${e}_all_max=[];\n L${layr}_E${e}_all_all=[];\n L${layr}_E${e}_all_avg=[];\n"
#set +x
#  done
  if [[ -z "$files" ]]; then
     continue
  fi
  echo $i has $files
  echo
  for j in $files
  do
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
               continue
            fi
        fi
            epc=`grep "Epochs in Training" $i/$j | awk -F:  '{print $2}'| tr -s " " | tr -d " "`
            if [ -z "$epc" ]; then
               epc=`grep "Training epochs"  $i/$j | awk '{ print $NF }'`
               if [ -z "$epc" ]; then
                  continue
               fi
            fi
            matefil=`grep "Build type is : $alttyp" $i/* | awk -F: '{print $1}' | sort -u`
            if [ -z "$matefil" ]; then
               continue
            elif ! grep "Training epochs set to $epc\|Epochs in Training : $epc" $matefil; then
               continue
            fi
            fil=$i/${type}_E${epc}_$j
            filout=${fil}.csv
            mv -v $i/$j $fil
            if [ -f "$filout" ]; then
               rm $filout
            fi
            tot_corr=`grep "Total Correct" $fil | uniq | awk -F: '{ print $2 }'`
            tot_tim=`grep "Total Time" $fil | uniq | awk -F: '{ print $2 }' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            tot_trng_tim=`grep "Total Train Time" $fil | uniq | awk -F: '{ print $2 }' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            tot_test_tim=`grep "Total Test Time" $fil | uniq | awk -F: '{ print $2 }' | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'`
            network=$i
            prefix1="L${layr}_E${epc}_${type}"
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
               prefix2="L${layr}_E${epc}_${comp1}"
               prefix3="L${layr}_E${epc}_${comp2}"
               altprefix1="L${layr}_E${epc}_Parallel"
               echo -e "${prefix1}_tot_time=$tot_tim;\n L${layr}_E${epc}_tot_PtoS=${prefix1}_tot_time/${altprefix1}_tot_time; \n  L${layr}_E${epc}_tot_PdiffS=${prefix1}_tot_time-${altprefix1}_tot_time; \n L${layr}_E${epc}_all_tots=[   L${layr}_E${epc}_tot_PtoS L${layr}_E${epc}_tot_PdiffS];\n" >> $filout
               echo -e "${prefix1}_tot_train=$tot_trng_tim; \n L${layr}_E${epc}_trg_PtoS=${prefix1}_tot_train/${altprefix1}_tot_train; \n  L${layr}_E${epc}_trg_PdiffS=${prefix1}_tot_train-${altprefix1}_tot_train; \n L${layr}_E${epc}_all_trgs=[   L${layr}_E${epc}_trg_PtoS L${layr}_E${epc}_trg_PdiffS];\n" >> $filout
               echo -e "${prefix1}_tot_test=$tot_test_tim; \n  L${layr}_E${epc}_tst_PtoS=${prefix1}_tot_test/${altprefix1}_tot_test; \n   L${layr}_E${epc}_tst_PdiffS=${prefix1}_tot_test-${altprefix1}_tot_test;\n   L${layr}_E${epc}_all_tsts=[   L${layr}_E${epc}_tst_PtoS L${layr}_E${epc}_tst_PdiffS];\n " >> $filout
            else
               echo "${prefix1}_tot_time=$tot_tim;" >> $filout
               echo "${prefix1}_tot_train=$tot_trng_tim;" >> $filout
               echo "${prefix1}_tot_test=$tot_test_tim;" >> $filout
            fi
     for k in $procs
     do
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
                maxt2="${pref}_max=$maxt;\n  ${pref2}_max=$maxt/${altpref}_max ; \n     ${pref3}_max=$maxt-${altpref}_max ; \n L${layr}_E${epc}_all_max= [  ${pref2}_max ${pref3}_max ];\n  "
                allt2="${pref}_all=$allt; \n ${pref2}_all=$allt/${altpref}_all ;  \n    ${pref3}_all=$allt-${altpref}_all ;\n  L${layr}_E${epc}_all_all= [  ${pref2}_all ${pref3}_all ];\n  "
                avgt2="${pref}_avg=$allt; \n ${pref2}_avg=$allt/${altpref}_avg ;  \n    ${pref3}_avg=$allt-${altpref}_avg ;\n  L${layr}_E${epc}_all_avg= [ ${pref2}_avg ${pref3}_avg ]; \n "
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
            real=`tail -20 $fil | grep real | awk '{print $2}'`
            user=`tail -20 $fil |grep user | awk '{print $2}'`
            sys=`tail -20 $fil |grep sys | awk '{print $2}'`
            if [ -n "$real" ]; then
               echo  "${prefix1}_real=\"$real\";" >>  $filout
               echo  "${prefix1}_sys=\"$sys\";" >>  $filout
               echo  "${prefix1}_user=\"$user\";" >>  $filout
            fi
     fi
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
echo -n "max_ratios=" >> mfile.m
echo -e $maxratios >> mfile.m
echo "];" >> mfile.m
echo -n "max_diffs=" >> mfile.m
echo -e $maxdiffs>> mfile.m
echo "];" >> mfile.m
echo -n "all_ratios=" >> mfile.m
echo -e $allratios >> mfile.m
echo "];" >> mfile.m
echo -n "all_diffs=" >> mfile.m
echo -e $alldiffs>> mfile.m
echo "];" >> mfile.m
echo -n "avg_ratios=" >> mfile.m
echo -e $avgratios >> mfile.m
echo "];" >> mfile.m
echo -n "avg_diffs=" >> mfile.m
echo -e $avgdiffs>> mfile.m
echo "];" >> mfile.m
echo -n "names=" >> mfile.m
echo -e $names >> mfile.m
echo "};" >> mfile.m
#grep "=\[\]" mfile.m > defs
#sed -i '/=\[\]/d'  mfile.m
#cat defs mfile.m  > mfile2.m

#set | grep $prefix1
