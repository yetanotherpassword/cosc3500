procs="add_mat add_scalar mult_scalar piecewisemult set_diff2_piecewisemult3 set_matmult set_mult1_add2_mat set_mult1_add2_scalars set_transpose"
grep "\"784 " slurm-1*.out | awk -F\" '{print $2}' | tr " " "_" | sort -u | awk '{  dir=$1; print "if [ -d \x22"dir"\x22 ]; then"; print "  rm -rf "dir; print "fi"; print "mkdir "dir }' > doit
grep "\"784 " slurm-1* | sed "s/ /_/g" | awk -F: '{a=match($2,"\""); dir = substr($2,a+1,length($2)-a-2);  print "cp "$1" "dir}' >> doit
chmod +x doit
./doit
dirs=`grep "\"784 " slurm-1* | sed "s/ /_/g" | awk -F: '{a=match($2,"\""); dir = substr($2,a+1,length($2)-a-2); print dir}'| sort -u`
#set -x
for i in $dirs
do
  files=`ls $i`
  echo $i has $files
  echo
  for j in $files
  do
     if grep -q "Killed" $i/$j ; then
         rm -v $i/$j
     fi
     if [ -f $i/$j ]; then
        type=`grep "Build type is :" $i/$j | awk -F: '{print $2}'| tr -s " " | tr -d " "`
        if [ -z "$type" ]; then
            echo "Build type not found in $i/$j, try older ref"
            if grep -q -i serial $i/$j; then
               type="Serial"
            elif  grep -q -i parallel $i/$j; then
               type="Parallel"
            else
               type="Unknown"
            fi
        else
            epc=`grep "Epochs in Training" $i/$j | awk -F:  '{print $2}'| tr -s " " | tr -d " "`
            if [ -z "$epc" ]; then
               epc=`grep "Training epochs"  $i/$j | awk '{ print $NF }'`
               if [ -z "$epc" ]; then
                  epc="EUNK"
               fi
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
            layr=`echo $i | awk -F_ '{print NF}'`
            prefix1="L${layr}_E${epc}_${type}"
            echo "Type=\"$type\"" >> $filout
            echo "${prefix1}_tot_corr=\"$tot_corr\"" >> $filout
            echo "${prefix1}_network=\"$network\"" >> $filout
            echo "Layers=$layr" >> $filout
            echo "Epochs="$epc >> $filout
            echo "${prefix1}_tot_time=$tot_tim" >> $filout
            echo "${prefix1}_tot_train=$tot_trng_tim" >> $filout
            echo "${prefix1}_tot_test=$tot_test_tim" >> $filout
     for k in $procs
     do
            grep -A 6 $k $fil > $i/tmp
            maxt=`grep Max $i/tmp | awk -F: '{print $2}'`
            allt=`grep All $i/tmp | awk -F: '{print $2}'`
            avgt=`grep Avg $i/tmp | awk -F: '{print $2}'`
            callsnum=`grep Calls $i/tmp | awk -F: '{print $2}'`
            pref="${prefix1}_${k}"
            echo ${pref}_max=$maxt | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'>> $filout
            echo ${pref}_avg=$avgt | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'>> $filout
            echo ${pref}_all=$allt | sed 's/ ns/\/1000000000/g; s/ ms/\/1000/g'>> $filout
            echo ${pref}_calls=$callsnum  >> $filout
     done
            real=`grep real $fil|grep -v =| awk '{print $2}'`
            user=`grep user $fil|grep -v =| awk '{print $2}'`
            sys=`grep sys $fil|grep -v =| awk '{print $2}'`
            echo  "${prefix1}_real=\"$real\"" >>  $filout
            echo  "${prefix1}_sys=\"$sys\"" >>  $filout
            echo  "${prefix1}_user=\"$user\"" >>  $filout
        fi
     fi
  done
done
set +x
