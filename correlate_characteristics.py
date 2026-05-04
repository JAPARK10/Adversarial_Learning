import numpy as np
from scipy.stats import pearsonr
import re

# 1. Raw Data from the user's logs
accuracy_log = """
test=p01 val=p02: acc=0.7548 f1=0.7472 auc=0.9803
test=p01 val=p03: acc=0.7643 f1=0.7505 auc=0.9815
test=p01 val=p04: acc=0.7286 f1=0.7170 auc=0.9863
test=p01 val=p05: acc=0.7476 f1=0.7353 auc=0.9820
test=p01 val=p06: acc=0.7286 f1=0.7140 auc=0.9790
test=p01 val=p07: acc=0.7381 f1=0.7262 auc=0.9873
test=p01 val=p08: acc=0.7548 f1=0.7410 auc=0.9814
test=p01 val=p09: acc=0.7381 f1=0.7313 auc=0.9773
test=p01 val=p10: acc=0.7500 f1=0.7415 auc=0.9813
test=p01 val=p11: acc=0.7429 f1=0.7328 auc=0.9860
test=p01 val=p12: acc=0.7548 f1=0.7498 auc=0.9848
test=p01 val=p13: acc=0.7476 f1=0.7357 auc=0.9808
test=p01 val=p14: acc=0.7738 f1=0.7652 auc=0.9850
test=p01 val=p15: acc=0.6881 f1=0.6764 auc=0.9798
test=p01 val=p16: acc=0.7929 f1=0.7865 auc=0.9875
test=p02 val=p01: acc=0.4619 f1=0.4507 auc=0.9330
test=p02 val=p03: acc=0.4857 f1=0.4870 auc=0.9139
test=p02 val=p04: acc=0.5024 f1=0.4936 auc=0.9250
test=p02 val=p05: acc=0.4786 f1=0.4689 auc=0.9144
test=p02 val=p06: acc=0.4952 f1=0.4741 auc=0.9270
test=p02 val=p07: acc=0.4905 f1=0.4746 auc=0.9364
test=p02 val=p08: acc=0.5143 f1=0.5019 auc=0.9450
test=p02 val=p09: acc=0.5238 f1=0.5128 auc=0.9424
test=p02 val=p10: acc=0.5024 f1=0.4984 auc=0.9215
test=p02 val=p11: acc=0.5000 f1=0.4869 auc=0.9341
test=p02 val=p12: acc=0.4929 f1=0.4792 auc=0.9397
test=p02 val=p13: acc=0.4786 f1=0.4779 auc=0.9235
test=p02 val=p14: acc=0.4952 f1=0.4768 auc=0.9331
test=p02 val=p15: acc=0.5071 f1=0.4966 auc=0.9404
test=p02 val=p16: acc=0.4762 f1=0.4595 auc=0.9399
test=p03 val=p01: acc=0.6357 f1=0.6254 auc=0.9619
test=p03 val=p02: acc=0.6310 f1=0.6187 auc=0.9645
test=p03 val=p04: acc=0.6262 f1=0.6144 auc=0.9632
test=p03 val=p05: acc=0.6286 f1=0.6149 auc=0.9542
test=p03 val=p06: acc=0.6190 f1=0.6082 auc=0.9618
test=p03 val=p07: acc=0.6286 f1=0.6074 auc=0.9612
test=p03 val=p08: acc=0.6167 f1=0.6176 auc=0.9614
test=p03 val=p09: acc=0.5810 f1=0.5709 auc=0.9514
test=p03 val=p10: acc=0.6048 f1=0.5963 auc=0.9604
test=p03 val=p11: acc=0.5714 f1=0.5656 auc=0.9529
test=p03 val=p12: acc=0.5976 f1=0.5885 auc=0.9599
test=p03 val=p13: acc=0.6095 f1=0.6035 auc=0.9596
test=p03 val=p14: acc=0.6310 f1=0.6168 auc=0.9631
test=p03 val=p15: acc=0.6095 f1=0.5986 auc=0.9634
test=p03 val=p16: acc=0.6262 f1=0.6029 auc=0.9655
test=p04 val=p01: acc=0.7286 f1=0.7162 auc=0.9827
test=p04 val=p02: acc=0.7286 f1=0.7114 auc=0.9820
test=p04 val=p03: acc=0.7905 f1=0.7809 auc=0.9889
test=p04 val=p05: acc=0.7357 f1=0.7241 auc=0.9827
test=p04 val=p06: acc=0.7643 f1=0.7516 auc=0.9838
test=p04 val=p07: acc=0.7357 f1=0.7226 auc=0.9840
test=p04 val=p08: acc=0.7524 f1=0.7414 auc=0.9853
test=p04 val=p09: acc=0.7381 f1=0.7196 auc=0.9875
test=p04 val=p10: acc=0.7929 f1=0.7881 auc=0.9901
test=p04 val=p11: acc=0.7429 f1=0.7314 auc=0.9875
test=p04 val=p12: acc=0.7429 f1=0.7292 auc=0.9874
test=p04 val=p13: acc=0.7238 f1=0.7070 auc=0.9844
test=p04 val=p14: acc=0.7548 f1=0.7311 auc=0.9858
test=p04 val=p15: acc=0.7357 f1=0.7208 auc=0.9850
test=p04 val=p16: acc=0.7595 f1=0.7409 auc=0.9860
test=p05 val=p01: acc=0.6095 f1=0.5768 auc=0.9523
test=p05 val=p02: acc=0.5333 f1=0.4879 auc=0.9415
test=p05 val=p03: acc=0.5738 f1=0.5483 auc=0.9592
test=p05 val=p04: acc=0.4952 f1=0.4629 auc=0.9423
test=p05 val=p06: acc=0.6024 f1=0.5655 auc=0.9443
test=p05 val=p07: acc=0.6143 f1=0.5837 auc=0.9574
test=p05 val=p08: acc=0.6190 f1=0.5827 auc=0.9574
test=p05 val=p09: acc=0.5881 f1=0.5592 auc=0.9547
test=p05 val=p10: acc=0.6095 f1=0.5687 auc=0.9519
test=p05 val=p11: acc=0.6262 f1=0.5912 auc=0.9478
test=p05 val=p12: acc=0.5810 f1=0.5364 auc=0.9498
test=p05 val=p13: acc=0.5881 f1=0.5476 auc=0.9607
test=p05 val=p14: acc=0.5262 f1=0.4791 auc=0.9471
test=p05 val=p15: acc=0.4929 f1=0.4405 auc=0.9428
test=p05 val=p16: acc=0.5738 f1=0.5288 auc=0.9557
test=p06 val=p01: acc=0.6429 f1=0.6375 auc=0.9641
test=p06 val=p02: acc=0.6071 f1=0.6005 auc=0.9556
test=p06 val=p03: acc=0.6571 f1=0.6520 auc=0.9656
test=p06 val=p04: acc=0.6500 f1=0.6438 auc=0.9704
test=p06 val=p05: acc=0.6071 f1=0.6080 auc=0.9599
test=p06 val=p07: acc=0.6381 f1=0.6356 auc=0.9681
test=p06 val=p08: acc=0.5976 f1=0.5937 auc=0.9585
test=p06 val=p09: acc=0.6429 f1=0.6362 auc=0.9652
test=p06 val=p10: acc=0.5548 f1=0.5549 auc=0.9515
test=p06 val=p11: acc=0.6310 f1=0.6183 auc=0.9655
test=p06 val=p12: acc=0.6476 f1=0.6412 auc=0.9657
test=p06 val=p13: acc=0.6286 f1=0.6281 auc=0.9598
test=p06 val=p14: acc=0.6143 f1=0.6125 auc=0.9569
test=p06 val=p15: acc=0.6405 f1=0.6251 auc=0.9525
test=p06 val=p16: acc=0.6381 f1=0.6256 auc=0.9685
test=p07 val=p01: acc=0.6762 f1=0.6621 auc=0.9678
test=p07 val=p02: acc=0.7048 f1=0.6839 auc=0.9729
test=p07 val=p03: acc=0.6357 f1=0.6067 auc=0.9612
test=p07 val=p04: acc=0.7119 f1=0.6958 auc=0.9697
test=p07 val=p05: acc=0.7190 f1=0.7063 auc=0.9667
test=p07 val=p06: acc=0.6833 f1=0.6638 auc=0.9621
test=p07 val=p08: acc=0.7310 f1=0.7145 auc=0.9716
test=p07 val=p09: acc=0.7071 f1=0.6938 auc=0.9676
test=p07 val=p10: acc=0.7119 f1=0.6964 auc=0.9658
test=p07 val=p11: acc=0.6929 f1=0.6775 auc=0.9703
test=p07 val=p12: acc=0.6762 f1=0.6583 auc=0.9656
test=p07 val=p13: acc=0.7119 f1=0.6859 auc=0.9725
test=p07 val=p14: acc=0.6976 f1=0.6758 auc=0.9618
test=p07 val=p15: acc=0.7095 f1=0.6909 auc=0.9564
test=p07 val=p16: acc=0.7071 f1=0.6842 auc=0.9701
test=p08 val=p01: acc=0.7152 f1=0.7049 auc=0.9823
test=p08 val=p02: acc=0.8030 f1=0.7975 auc=0.9901
test=p08 val=p03: acc=0.8030 f1=0.7933 auc=0.9932
test=p08 val=p04: acc=0.8333 f1=0.8306 auc=0.9886
test=p08 val=p05: acc=0.7727 f1=0.7590 auc=0.9887
test=p08 val=p06: acc=0.8212 f1=0.8204 auc=0.9917
test=p08 val=p07: acc=0.8121 f1=0.8049 auc=0.9869
test=p08 val=p09: acc=0.8030 f1=0.7971 auc=0.9900
test=p08 val=p10: acc=0.7939 f1=0.7935 auc=0.9867
test=p08 val=p11: acc=0.7788 f1=0.7714 auc=0.9839
test=p08 val=p12: acc=0.7818 f1=0.7760 auc=0.9912
test=p08 val=p13: acc=0.7667 f1=0.7557 auc=0.9897
test=p08 val=p14: acc=0.7576 f1=0.7456 auc=0.9894
test=p08 val=p15: acc=0.7727 f1=0.7657 auc=0.9909
test=p08 val=p16: acc=0.7727 f1=0.7651 auc=0.9890
test=p09 val=p01: acc=0.7643 f1=0.7491 auc=0.9829
test=p09 val=p02: acc=0.7214 f1=0.7046 auc=0.9840
test=p09 val=p03: acc=0.7548 f1=0.7313 auc=0.9822
test=p09 val=p04: acc=0.7738 f1=0.7561 auc=0.9847
test=p09 val=p05: acc=0.7667 f1=0.7417 auc=0.9785
test=p09 val=p06: acc=0.7619 f1=0.7466 auc=0.9821
test=p09 val=p07: acc=0.7667 f1=0.7581 auc=0.9843
test=p09 val=p08: acc=0.7952 f1=0.7844 auc=0.9841
test=p09 val=p10: acc=0.7690 f1=0.7545 auc=0.9855
test=p09 val=p11: acc=0.7833 f1=0.7590 auc=0.9845
test=p09 val=p12: acc=0.7643 f1=0.7522 auc=0.9835
test=p09 val=p13: acc=0.7786 f1=0.7649 auc=0.9862
test=p09 val=p14: acc=0.8119 f1=0.8009 auc=0.9831
test=p09 val=p15: acc=0.7619 f1=0.7497 auc=0.9856
test=p09 val=p16: acc=0.7857 f1=0.7773 auc=0.9869
test=p10 val=p01: acc=0.6905 f1=0.6885 auc=0.9821
test=p10 val=p02: acc=0.7000 f1=0.6945 auc=0.9818
test=p10 val=p03: acc=0.7476 f1=0.7400 auc=0.9862
test=p10 val=p04: acc=0.7024 f1=0.6944 auc=0.9843
test=p10 val=p05: acc=0.7667 f1=0.7634 auc=0.9878
test=p10 val=p06: acc=0.7214 f1=0.7200 auc=0.9830
test=p10 val=p07: acc=0.6929 f1=0.6893 auc=0.9802
test=p10 val=p08: acc=0.7571 f1=0.7537 auc=0.9860
test=p10 val=p09: acc=0.6548 f1=0.6526 auc=0.9807
test=p10 val=p11: acc=0.7500 f1=0.7497 auc=0.9824
test=p10 val=p12: acc=0.7548 f1=0.7499 auc=0.9849
test=p10 val=p13: acc=0.7262 f1=0.7153 auc=0.9831
test=p10 val=p14: acc=0.6929 f1=0.6899 auc=0.9829
test=p10 val=p15: acc=0.7881 f1=0.7855 auc=0.9876
test=p10 val=p16: acc=0.7357 f1=0.7302 auc=0.9844
test=p11 val=p01: acc=0.5952 f1=0.5743 auc=0.9596
test=p11 val=p02: acc=0.6595 f1=0.6446 auc=0.9677
test=p11 val=p03: acc=0.6643 f1=0.6507 auc=0.9733
test=p11 val=p04: acc=0.6476 f1=0.6372 auc=0.9695
test=p11 val=p05: acc=0.6571 f1=0.6372 auc=0.9678
test=p11 val=p06: acc=0.6143 f1=0.6008 auc=0.9714
test=p11 val=p07: acc=0.6167 f1=0.6119 auc=0.9664
test=p11 val=p08: acc=0.5929 f1=0.5817 auc=0.9596
test=p11 val=p09: acc=0.6524 f1=0.6315 auc=0.9721
test=p11 val=p10: acc=0.6524 f1=0.6415 auc=0.9680
test=p11 val=p12: acc=0.6476 f1=0.6360 auc=0.9710
test=p11 val=p13: acc=0.6190 f1=0.6061 auc=0.9592
test=p11 val=p14: acc=0.5690 f1=0.5408 auc=0.9505
test=p11 val=p15: acc=0.5857 f1=0.5715 auc=0.9641
test=p11 val=p16: acc=0.6048 f1=0.5891 auc=0.9604
test=p12 val=p01: acc=0.7595 f1=0.7436 auc=0.9816
test=p12 val=p02: acc=0.7214 f1=0.7142 auc=0.9818
test=p12 val=p03: acc=0.7024 f1=0.6721 auc=0.9793
test=p12 val=p04: acc=0.7405 f1=0.7265 auc=0.9804
test=p12 val=p05: acc=0.7762 f1=0.7671 auc=0.9878
test=p12 val=p06: acc=0.6952 f1=0.6729 auc=0.9760
test=p12 val=p07: acc=0.7690 f1=0.7423 auc=0.9838
test=p12 val=p08: acc=0.7238 f1=0.7054 auc=0.9823
test=p12 val=p09: acc=0.6833 f1=0.6609 auc=0.9793
test=p12 val=p10: acc=0.7548 f1=0.7386 auc=0.9827
test=p12 val=p11: acc=0.7810 f1=0.7600 auc=0.9851
test=p12 val=p13: acc=0.7095 f1=0.6935 auc=0.9820
test=p12 val=p14: acc=0.7595 f1=0.7375 auc=0.9818
test=p12 val=p15: acc=0.7381 f1=0.7178 auc=0.9767
test=p12 val=p16: acc=0.7286 f1=0.7105 auc=0.9798
test=p13 val=p01: acc=0.5524 f1=0.5323 auc=0.9365
test=p13 val=p02: acc=0.5929 f1=0.5842 auc=0.9348
test=p13 val=p03: acc=0.4833 f1=0.4444 auc=0.9472
test=p13 val=p04: acc=0.5310 f1=0.5005 auc=0.9364
test=p13 val=p05: acc=0.6238 f1=0.6037 auc=0.9454
test=p13 val=p06: acc=0.5833 f1=0.5517 auc=0.9485
test=p13 val=p07: acc=0.6381 f1=0.6213 auc=0.9589
test=p13 val=p08: acc=0.6143 f1=0.6026 auc=0.9436
test=p13 val=p09: acc=0.6214 f1=0.6024 auc=0.9529
test=p13 val=p10: acc=0.5500 f1=0.5159 auc=0.9443
test=p13 val=p11: acc=0.5357 f1=0.5066 auc=0.9258
test=p13 val=p12: acc=0.6381 f1=0.6199 auc=0.9615
test=p13 val=p14: acc=0.5976 f1=0.5754 auc=0.9587
test=p13 val=p15: acc=0.5357 f1=0.5033 auc=0.9411
test=p13 val=p16: acc=0.5452 f1=0.5188 auc=0.9346
test=p14 val=p01: acc=0.5111 f1=0.4865 auc=0.9576
test=p14 val=p02: acc=0.5397 f1=0.5189 auc=0.9402
test=p14 val=p03: acc=0.5556 f1=0.5407 auc=0.9569
test=p14 val=p04: acc=0.5175 f1=0.5059 auc=0.9576
test=p14 val=p05: acc=0.5556 f1=0.5485 auc=0.9733
test=p14 val=p06: acc=0.5937 f1=0.5757 auc=0.9665
test=p14 val=p07: acc=0.4857 f1=0.4555 auc=0.9554
test=p14 val=p08: acc=0.6413 f1=0.6209 auc=0.9715
test=p14 val=p09: acc=0.6032 f1=0.5856 auc=0.9624
test=p14 val=p10: acc=0.5746 f1=0.5546 auc=0.9664
test=p14 val=p11: acc=0.5524 f1=0.5052 auc=0.9425
test=p14 val=p12: acc=0.6127 f1=0.5707 auc=0.9640
test=p14 val=p13: acc=0.5556 f1=0.5261 auc=0.9619
test=p14 val=p15: acc=0.5048 f1=0.4691 auc=0.9599
test=p14 val=p16: acc=0.7333 f1=0.6975 auc=0.9817
test=p15 val=p01: acc=0.6738 f1=0.6574 auc=0.9776
test=p15 val=p02: acc=0.6786 f1=0.6490 auc=0.9756
test=p15 val=p03: acc=0.6500 f1=0.6222 auc=0.9678
test=p15 val=p04: acc=0.6643 f1=0.6433 auc=0.9774
test=p15 val=p05: acc=0.6857 f1=0.6651 auc=0.9785
test=p15 val=p06: acc=0.6905 f1=0.6694 auc=0.9775
test=p15 val=p07: acc=0.7119 f1=0.6874 auc=0.9821
test=p15 val=p08: acc=0.7024 f1=0.6887 auc=0.9822
test=p15 val=p09: acc=0.6857 f1=0.6656 auc=0.9814
test=p15 val=p10: acc=0.7286 f1=0.7143 auc=0.9839
test=p15 val=p11: acc=0.6905 f1=0.6702 auc=0.9788
test=p15 val=p12: acc=0.6714 f1=0.6460 auc=0.9795
test=p15 val=p13: acc=0.7190 f1=0.7059 auc=0.9874
test=p15 val=p14: acc=0.6452 f1=0.6056 auc=0.9707
test=p15 val=p16: acc=0.7357 f1=0.7226 auc=0.9837
test=p16 val=p01: acc=0.3175 f1=0.2945 auc=0.8309
test=p16 val=p02: acc=0.2730 f1=0.2300 auc=0.8171
test=p16 val=p03: acc=0.2349 f1=0.2097 auc=0.8197
test=p16 val=p04: acc=0.2921 f1=0.2583 auc=0.8353
test=p16 val=p05: acc=0.2889 f1=0.2571 auc=0.8364
test=p16 val=p06: acc=0.3492 f1=0.3112 auc=0.8430
test=p16 val=p07: acc=0.2921 f1=0.2644 auc=0.8351
test=p16 val=p08: acc=0.2762 f1=0.2304 auc=0.8382
test=p16 val=p09: acc=0.3619 f1=0.3288 auc=0.8345
test=p16 val=p10: acc=0.3206 f1=0.2830 auc=0.8293
test=p16 val=p11: acc=0.3079 f1=0.2809 auc=0.8345
test=p16 val=p12: acc=0.3143 f1=0.2768 auc=0.8443
test=p16 val=p13: acc=0.3302 f1=0.2980 auc=0.8617
test=p16 val=p14: acc=0.3238 f1=0.3033 auc=0.8404
test=p16 val=p15: acc=0.3524 f1=0.3184 auc=0.8654
"""

# 2. Characteristics from the Deep Scan script
characteristics = {
    1:  {'speed': 0.1906, 'noise': 0.0291, 'bias': 0.3659},
    2:  {'speed': 0.1873, 'noise': 0.0266, 'bias': 0.5387},
    3:  {'speed': 0.1875, 'noise': 0.0239, 'bias': 0.4355},
    4:  {'speed': 0.1808, 'noise': 0.0270, 'bias': 0.2738},
    5:  {'speed': 0.1910, 'noise': 0.0233, 'bias': 0.3474},
    6:  {'speed': 0.1679, 'noise': 0.0209, 'bias': 0.3365},
    7:  {'speed': 0.1389, 'noise': 0.0240, 'bias': 0.3669},
    8:  {'speed': 0.1655, 'noise': 0.0280, 'bias': 0.3477},
    9:  {'speed': 0.1709, 'noise': 0.0218, 'bias': 0.3573},
    10: {'speed': 0.1595, 'noise': 0.0312, 'bias': 0.3392},
    11: {'speed': 0.1952, 'noise': 0.0251, 'bias': 0.3218},
    12: {'speed': 0.1628, 'noise': 0.0322, 'bias': 0.4471},
    13: {'speed': 0.1606, 'noise': 0.0263, 'bias': 0.3481},
    14: {'speed': 0.1929, 'noise': 0.0216, 'bias': 0.5098},
    15: {'speed': 0.1639, 'noise': 0.0217, 'bias': 0.3593},
    16: {'speed': 0.1688, 'noise': 0.0220, 'bias': 0.4531},
}

def analyze_correlations():
    # Parse Accuracies
    person_accs = {}
    pattern = re.compile(r"test=p(\d+).+acc=([\d.]+)")
    
    for line in accuracy_log.strip().split('\n'):
        match = pattern.search(line)
        if match:
            pid = int(match.group(1))
            acc = float(match.group(2))
            if pid not in person_accs: person_accs[pid] = []
            person_accs[pid].append(acc)
            
    # Calculate Averages
    averages = {pid: np.mean(accs) for pid, accs in person_accs.items()}
    
    # Prepare vectors for correlation
    pids = sorted(averages.keys())
    y_acc = [averages[pid] for pid in pids]
    x_speed = [characteristics[pid]['speed'] for pid in pids]
    x_noise = [characteristics[pid]['noise'] for pid in pids]
    x_bias  = [characteristics[pid]['bias'] for pid in pids]
    
    print("\n" + "="*60)
    print(f"{'PID':<6} | {'AVG ACC':<10} | {'SPEED':<10} | {'NOISE':<10} | {'BIAS'}")
    print("-" * 60)
    for i, pid in enumerate(pids):
        print(f"P{pid:02d}    | {y_acc[i]:.4f}     | {x_speed[i]:.4f}     | {x_noise[i]:.4f}     | {x_bias[i]:.4f}")
    print("="*60)
    
    # Pearson Correlation
    corr_speed, _ = pearsonr(x_speed, y_acc)
    corr_noise, _ = pearsonr(x_noise, y_acc)
    corr_bias,  _  = pearsonr(x_bias,  y_acc)
    
    print("\nPEARSON CORRELATION (Characteristic vs. Accuracy):")
    print(f"  * Speed Correlation: {corr_speed:+.4f}")
    print(f"  * Noise Correlation: {corr_noise:+.4f}")
    print(f"  * Bias Correlation:  {corr_bias:+.4f}")
    
    print("\nINTERPRETATION:")
    if abs(corr_speed) > 0.4:
        direction = "Positive" if corr_speed > 0 else "Negative"
        print(f"  - {direction} relationship found with Speed. Moving {'faster' if corr_speed > 0 else 'slower'} helps accuracy.")
    if abs(corr_noise) > 0.4:
        direction = "Positive" if corr_noise > 0 else "Negative"
        print(f"  - {direction} relationship found with Noise. Clean signals are {'better' if corr_noise < 0 else 'surprisingly harder'}.")
    if abs(corr_bias) > 0.4:
        direction = "Positive" if corr_bias > 0 else "Negative"
        print(f"  - {direction} relationship found with Spatial Bias. Being off-center {'helps' if corr_bias > 0 else 'hurts'} the model.")

if __name__ == "__main__":
    analyze_correlations()
