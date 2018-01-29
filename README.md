# generate-jokes
This repository will host my work on generating jokes using the r/jokes corpus. It is a follow up to my capstone project for Udacity's Machine Learning Nanodegree. The ultimate goal is to deploy it as either a twitter or reddit bot.

## Topics
The first script/notebook generates topic vectors using non-negative matrix factorization. The topic vectors have 32 dimensions.

The following table diplays the the top 20 most influential words per topic dimension. Warning, it contains some vulgar words.

|Dimension | Words |
|----------|-------|
0|say,ask,prostitute,look,leper,tampon,reply,turn,stick,tip,blonde,fish,nice,think,little,pirate,boy,friend,thing,drive
1|like,coffee,slave,penis,free,chocolate,life,dark,taste,smell,dick,food,look,humor,feel,box,lot,sound,beer,domestic
2|joke,funny,punchline,dick,laugh,mom,bad,reddit,apparent,post,r,unemployed,work,chemistry,wanna,racist,title,long,want,execution
3|lightbulb,change,feminist,screw,alzheimer,funny,dark,juan,mexicans,basement,patient,pregnant,room,number,germans,beat,efficient,optometrist,trick,dead
4|bar,walk,bartender,drink,beer,serve,order,duck,ask,table,irishman,chair,past,bear,blonde,tense,termite,future,horse,roman
5|knock,door,forget,prize,freedom,bell,mop,america,daisy,invent,ring,dish,sally,jehovah,favorite,alzheimer,interrupt,person,pencil,witness
6|people,hate,world,type,fat,kind,die,threesome,talk,count,unemployed,mean,binary,condescend,work,understand,life,think,lot,chocolate
7|man,reply,ask,park,old,warm,day,car,look,chinese,jewish,life,spaceman,second,fish,sit,rest,run,young,space
8|difference,s,dick,jesus,outlaw,mom,job,prostitute,snowman,hooker,porcupine,thermometer,picture,face,hitler,suck,feminist,ignorance,acne,irish
9|hear,want,kidnapping,new,mexican,pencil,mathematician,work,calendar,wanna,steal,deaf,constipated,pterodactyl,wake,probably,hipster,hole,die,circus
10|girlfriend,dump,cannibal,wipe,break,laugh,ex,imaginary,competitive,ass,think,fat,start,fit,homeless,clothe,smoke,relationship,pissed,slow
11|dad,son,father,boy,mom,kid,ask,mother,reply,dollar,little,want,school,jewish,johnny,masturbate,daddy,adopt,daughter,parent
12|make,atom,trust,water,boil,holy,hell,hebrew,hormone,moses,tea,ugly,day,tickle,money,pay,love,anal,hole,weak
13|use,condom,word,steal,addict,teacher,bear,soap,clean,time,indecisive,goodyear,tire,bike,hand,common,hate,parachute,think,circumcision
14|know,letter,alphabet,nose,ignorance,really,orphan,apathy,body,care,feel,baseball,number,play,ladder,shoe,drive,drug,lace,mean
15|dog,leg,zoo,leave,labracadabrador,matter,cat,come,shitzu,bike,right,short,dyslexic,magic,arm,bleed,skydive,agnostic,night,little
16|girl,date,boy,meet,little,ask,number,want,attractive,fat,today,home,damn,odd,drop,homeless,hey,night,tit,eventually
17|sex,fuck,anal,time,life,oral,common,threesome,phone,camp,position,hole,object,tent,ask,weak,try,mate,number,chinese
18|woman,ugly,husband,jewish,wrong,easy,time,pregnant,pick,coffee,driver,heavy,think,beautiful,love,ask,money,drive,baby,inch
19|black,white,racist,afraid,cop,society,trump,paint,beat,police,shoot,fall,cruise,stair,person,work,mexican,room,batman,run
20|cross,chicken,road,kill,door,egg,sedan,sock,wrong,coop,mexican,wear,potato,favorite,jesus,sperm,semen,diana,princess,titanic
21|doctor,patient,bad,news,masturbate,stop,live,doc,ask,examine,alzheimer,problem,try,month,reply,exam,cancer,office,prostate,need
22|good,thing,bad,really,tree,hide,elephant,friend,news,period,time,way,day,pun,ruin,piano,organ,miss,s,rose
23|cow,beef,leg,milk,ground,masturbate,mad,disease,jerky,hoof,farmer,field,stroganoff,udder,pms,count,abortion,birth,lactose,lip
24|light,bulb,screw,change,hard,sleep,fly,optometrist,cop,room,heavy,zippo,alzheimer,hippo,beat,cigarette,blue,patient,dark,psychiatrist
25|guy,come,gay,steal,think,calendar,second,hitler,look,drunk,happen,right,month,wish,hey,left,fuck,ask,kill,stop
26|just,think,kid,work,today,job,day,word,new,lose,want,buy,really,break,trump,ice,need,car,plagiarism,watch
27|tell,friend,laugh,ask,stop,time,vegan,worry,teacher,blonde,day,mean,high,boss,want,try,parent,surprised,eyebrow,draw
28|eat,gay,horse,cannibal,vegetable,hard,wheelchair,lesbian,common,time,dinosaur,clock,shit,zombie,consume,pizza,fish,taste,vegetarian,food
29|wife,husband,want,die,home,happy,ask,night,meet,leave,love,sleep,car,house,job,bed,fat,ex,honey,cheat
30|year,old,come,santa,sack,lady,big,today,vision,happy,new,live,claus,ago,meet,time,common,kid,child,day
31|blind,fall,german,hard,eye,table,chair,skydive,dinosaur,step,nudist,scar,lady,date,tree,prostitute,deer,beach,colony,spot
