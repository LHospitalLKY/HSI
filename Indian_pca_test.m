clear
load 'D:\�߹���\�߹�������\�߹�������\Indian\Indian_pines'
load 'D:\�߹���\�߹�������\�߹�������\Indian\Indian_pines_gt'
indian_pines_vector = reshape(indian_pines,145^2,220);%��������Ϊһ��145^2�У�220�еľ���
indian_pines_gt_vector = reshape(indian_pines_gt,145^2,1);%���ܱ��
indian_pines_vector(find(indian_pines_gt_vector == 0),:) = [];%ɾ�����Ϊ0�ĵ㣬��������
indian_pines_gt_vector(find(indian_pines_gt_vector == 0)) = [];%ɾ���������

[coeff,score,latent] = pca(indian_pines_vector);%PCA��ά����coeffΪת������latent����ת���ɹ�������Ϣ
sum_0 = 0;
r = zeros(220,1);
for k = 1:220                            %
    sum_0 = sum_0 + latent(k);
    r(k) = sum_0 / sum(latent);
end
n_prin = find(r <= 0.99);                %��ά
indian_pines_lowdimension = indian_pines_vector*coeff(:,n_prin);
indian_pines_lowdimension = indian_pines_lowdimension';

indian_pines_lowdimension_label_1 = indian_pines_lowdimension(:,find(indian_pines_gt_vector == 3));
indian_pines_lowdimension_label_2 = indian_pines_lowdimension(:,find(indian_pines_gt_vector == 6));
[~,n_1] = size(indian_pines_lowdimension_label_1);
[~,n_2] = size(indian_pines_lowdimension_label_2);

for k = 1:1:2
    p = 0.1 * k;
    parfor l = 1 : 1 : 100
        lambda = l * 0.01;
        for m = 1:1:10
            epsilon = m * 0.1;
            
            %Construct the train database
            train_1 = randperm(n_1,floor(0.1*n_1));
            train_2 = randperm(n_2,floor(0.1*n_2));
            indian_pines_lowdimension_label_1_train = indian_pines_lowdimension_label_1(:,train_1);
            indian_pines_lowdimension_label_2_train = indian_pines_lowdimension_label_2(:,train_2);
            %Construct the test database
            indian_pines_lowdimension_label_1_test = indian_pines_lowdimension_label_1;
            indian_pines_lowdimension_label_2_test = indian_pines_lowdimension_label_2;
            indian_pines_lowdimension_label_1_test(:, train_1) = [];
            indian_pines_lowdimension_label_2_test(:, train_2) = [];
            indian_pines_lowdimension_test = [indian_pines_lowdimension_label_1_test indian_pines_lowdimension_label_2_test];
            
            for i = 1 : floor(0.1*n_1)
                [ output_args ] = p_contral_CRC( indian_pines_lowdimension_label_1_train(:, i), indian_pines_lowdimension_test, p, lambda, epsilon );
            end
        end
    end
end

