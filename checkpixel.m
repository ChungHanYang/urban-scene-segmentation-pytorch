PathRoot='/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/train/train3/';
list=dir(fullfile(PathRoot));
car = 0;
person = 0;
road = 0;
sidewalk = 0;
building = 0;
wall = 0;
fence = 0;
pole = 0;
traffic_light = 0;
traffic_sign = 0;
vegetation = 0;
terrain = 0 ;
sky = 0;
rider = 0;
truck = 0;
bus = 0;
train = 0;
motorcycle = 0;
bicycle = 0;
background =0;
for k=3:size(list,1)
     
    filename = ['/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/train/train3/',list(k).name];
    %outputfile = ['/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/val/val2/',list(k).name(1:36),'.mat'];
  
    load(filename,'y');
    
    for i=1:512
        for j=1:1024
            if y(i,j)==0
                car = car+1;
            elseif y(i,j)==1
                person = person+1;
            elseif y(i,j)==2
                road = road+1;
            elseif y(i,j)==3
                sidewalk = sidewalk+1;
            elseif y(i,j)==4
                building = building+1;
            elseif y(i,j)==5
                wall = wall+1;
            elseif y(i,j)==6
                fence = fence+1;
            elseif y(i,j)==7
                pole = pole+1;
            elseif y(i,j)==8
                traffic_light = traffic_light+1;
            elseif y(i,j)==9
                traffic_sign = traffic_sign+1;
            elseif y(i,j)==10
                vegetation = vegetation+1;
            elseif y(i,j)==11
                terrain = terrain+1;
            elseif y(i,j)==12
                sky = sky+1;
            elseif y(i,j)==13
                rider = rider+1;
            elseif y(i,j)==14
                truck = truck+1;
            elseif y(i,j)==15
                bus = bus+1;
            elseif y(i,j)==16
                train = train+1;
            elseif y(i,j)==17
                motorcycle = motorcycle+1;
            elseif y(i,j)==18
                bicycle = bicycle+1;
            elseif y(i,j)==19
                background = background+1;
                     
               
               
            
            end
        end
    end
    
    
    
end
total = car+person+road+sidewalk+background+bicycle+building+bus+fence+motorcycle+pole+rider+sky+terrain+traffic_light+traffic_sign+train+truck+vegetation+wall;
% car = car/total;
% person = person/total;
% background = background/total;
% road = road/total;
% rider = rider/total;
