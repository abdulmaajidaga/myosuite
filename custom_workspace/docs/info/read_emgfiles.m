[fname, pname] = uigetfile('*.csv', 'Select data');
filename=fullfile(pname, fname);

data=csvread(filename)

muscle_set={'M. Deltoideus pars clavicularis', 'M. Biceps brachii', 'M.Triceps brachii', 'M. Flexor digitorum superficialis', 'M. Extensor digitorum', 'M. Brachioradialis', 'M. Flexor carpi ulnaris', 'M. Extensor carpi ulnaris', 'M. Pronator teres', 'M. Flexor carpi radialis', 'M. Abductor pollicis brevis', 'M. Abductor digit minimi'}; 

figure(1)
for i=1:12
subplot(3,4,i)
plot(data(:,i))
legend(muscle_set{1,i});
end
