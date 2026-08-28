% GulfMexico (Gulf of Mexico) extracted on May 30, 2026
% extract all fields for 2023_03_27_000000 to 2023_09_28_210000
% (example extraction on face 4 or 5, i.e., rotated UV fields)

% {{{ initialize
clear
addpath /nobackup/kzhang/llc_4320/extract/matlab_v0
load /nobackup/kzhang/llc_4320/extract/matlab_v0/offsets
clear pnu dte

%determine time stamp of available input files
cd /nobackupp27/dbwhitt/llc_4320/OUT
pin=dir('*/U*.shrunk');
tme=zeros(size(pin));
pnm=' ';
for t=1:length(tme)
    if strcmp(pnm,pin(t).folder)
        n=n+1;
    else
        pnm=pin(t).folder;
        n=0;
    end
    yr=str2num(pnm(35:38));
    mo=str2num(pnm(40:41));
    dy=str2num(pnm(43:44));
    hr=str2num(pnm(46:47));
    tme(t)=datenum(yr,mo,dy,hr,0,0)+n/24;
end
% discard times before 2023_03_27_000000
pin(find(tme<datenum(2023,3,27)))=[];
tme(find(tme<datenum(2023,3,27)))=[];
% }}}

% {{{ define desired region
prec='real*4';
minlat=18.1;
maxlat=30.74;
minlon=-97.91;
maxlon=-80.58;
region_name='GulfMexico';
pout=['/nobackupp28/kzhang/llc_4320/regions/' region_name '/'];
% }}}

% {{{ extract indices for desired region
fnam=[gridDir 'Depth.data'];
[fld fc ix jx] = ...
    quikread_llc(fnam,FS,1,prec,gridDir,minlat,maxlat,minlon,maxlon);
fld(find(fld==0))=nan;
xc=read_llc_fkij([gridDir 'XC.data'],FS,fc,1,ix,jx);
yc=read_llc_fkij([gridDir 'YC.data'],FS,fc,1,ix,jx);
pcolorcen(xc',yc',fld')
colorbar
RF=readbin([gridDir 'RF.data'],NZ+1);
bot=-RF(2:end);
kx=1:min(find(bot>mmax(fld)));
suf1=['_' int2str(length(ix)) 'x' int2str(length(jx))];
suf2=[suf1 'x' int2str(length(kx))];
close all
% }}}

% {{{ get and save grid information
eval(['mkdir ' pout 'grid'])
eval(['cd ' pout 'grid'])

% {{{ grid cell center
for fnm={'RAC','XC','YC','Depth','hFacC'}
    fin=[gridDir fnm{1} '.data'];
    switch fnm{1}
      case{'hFacC'}
        fld=read_llc_fkij(fin,FS,fc,kx,ix,jx);
        fout=[fnm{1} suf2];
      otherwise
        fld=read_llc_fkij(fin,FS,fc,1,ix,jx);
        fout=[fnm{1} suf1];
    end
    writebin(fout,fld);
end
% }}}

% {{{ AngleCS and AngleSN at grid cell centers
% need to be rotated because U/V vectors are rotated
fnx='AngleCS';
fny='AngleSN';
finx=[gridDir fnx '.data'];
finy=[gridDir fny '.data'];
foutx=[fnx suf1];
fouty=[fny suf1];
switch fc
  case {1,2}
    fldx=read_llc_fkij(finx,FS,fc,1,ix,jx);
    fldy=read_llc_fkij(finy,FS,fc,1,ix,jx);
  case {4,5}
    fldx=-read_llc_fkij(finy,FS,fc,1,ix,jx); % <<<<<<<<
    fldy=read_llc_fkij(finx,FS,fc,1,ix,jx);
end
writebin(foutx,fldx);
writebin(fouty,fldy);
% }}}

% {{{ Southwest corner (vorticity) points, no direction
for fnm={'XG','YG','RAZ'}
    fin=[gridDir fnm{1} '.data'];
    fld=read_llc_fkij(fin,FS,fc,1,ix,jx-1); % <<<<<<<<
    fout=[fnm{1} suf1];
    writebin(fout,fld);
end
% }}}

% {{{ West edge points, no direction
fnx='DXC';
fny='DYC';
finx=[gridDir fnx '.data'];
finy=[gridDir fny '.data'];
foutx=[fnx suf1];
fouty=[fny suf1];
fldx=read_llc_fkij(finy,FS,fc,1,ix,jx);
fldy=read_llc_fkij(finx,FS,fc,1,ix,jx-1); % <<<<<<<<
writebin(foutx,fldx);
writebin(fouty,fldy);
% }}}

% {{{ Southwest corner (vorticity) points, no direction
fnx='DXV';
fny='DYU';
finx=[gridDir fnx '.data'];
finy=[gridDir fny '.data'];
foutx=[fnx suf1];
fouty=[fny suf1];
fldx=read_llc_fkij(finy,FS,fc,1,ix,jx);
fldy=read_llc_fkij(finx,FS,fc,1,ix,jx-1); % <<<<<<<<
writebin(foutx,fldx);
writebin(fouty,fldy);
% }}}

% {{{ Southwest edge points, no direction
fnx='DXG';
fny='DYG';
finx=[gridDir fnx '.data'];
finy=[gridDir fny '.data'];
foutx=[fnx suf1];
fouty=[fny suf1];
fldx=read_llc_fkij(finy,FS,fc,1,ix,jx);
fldy=read_llc_fkij(finx,FS,fc,1,ix,jx);
writebin(foutx,fldx);
writebin(fouty,fldy);
% }}}

% }}}

% {{{ get model output

% {{{ get and save scalar 2D fields
for fnm={'KPPhbl','Eta','PhiBot','oceQnet','oceQsw','oceFWflx'}
    eval(['mkdir ' pout fnm{1}])
    eval(['cd ' pout fnm{1}])
    for t=1:length(tme)
        fin=[pin(t).folder '/' fnm{1} pin(t).name(2:end-6) 'data'];
        dy=datestr(tme(t),30);
        fout=[fnm{1} suf1 '.' dy];
        fld=read_llc_fkij(fin,FS,fc,1,ix,jx);
        writebin(fout,fld);
    end
end
% }}}

% {{{ get and save vector 2D fields
% note that zonal velocity is oceTAUX in faces 1/2 and oceTAUY in faces 4/5
% and meridional velocity is oceTAUY in faces 1/2 and -oceTAUX in faces 4/5
eval(['mkdir ' pout 'oceTAUX'])
eval(['mkdir ' pout 'oceTAUY'])
eval(['cd ' pout])
for t=1:length(tme)
    finu=[pin(t).folder '/oceTAUX' pin(t).name(2:end-6) 'data'];
    finv=[pin(t).folder '/oceTAUY' pin(t).name(2:end-6) 'data'];
    dy=datestr(tme(t),30);
    foutu=['oceTAUX/oceTAUX' suf1 '.' dy];
    foutv=['oceTAUY/oceTAUY' suf1 '.' dy];
    fldu=read_llc_fkij(finv,FS,fc,1,ix,jx);
    fldv=-read_llc_fkij(finu,FS,fc,1,ix,jx-1);
    writebin(foutu,fldu);
    writebin(foutv,fldv);
end
% }}}

% {{{ get and save scalar 3D fields
xRange=[3*FS+ix(1) length(ix)];
yRange=[jx(1) length(jx)];
zRange=[1 length(kx)];
maskFile=[maskDir 'hFacC.bits'];
for fnm={'Salt','Theta','W'}
    eval(['mkdir ' pout fnm{1}])
    eval(['cd ' pout fnm{1}])
    for t=1:length(tme)
        fin=[pin(t).folder '/' fnm{1} pin(t).name(2:end)];
        dy=datestr(tme(t),30);
        fout=[fnm{1} suf2 '.' dy];
        fld=llc_shrunk_read(fin,maskFile,FS,'xRange',xRange, ...
            'yRange',yRange,'zRange',zRange,'offsets',off_C);
        writebin(fout,fld);
    end
end
% }}}

% {{{ generate commands to extract scalar 3D fields
startPoint=[' ' int2str(3*FS+ix(1)) ',' int2str(jx(1)) ',1 '];
extent=[int2str(length(ix)) ',' int2str(length(jx)) ',' int2str(length(kx))];
for fnm={'Salt','Theta','W'}
    eval(['mkdir ' pout fnm{1}])
    eval(['cd ' pout fnm{1}])
    for t=1:length(tme)
        dy=datestr(tme(t),30);
        fout=[fnm{1} suf2 '.' dy];
        new_line=['~/llc_4320/extract/dan_v0/extract4320 -d ' pin(t).folder ' -m ' ...
                  '/nobackupp29/bcnelson/MITgcm/compressedOutput/work/bitmask.4320x173' ...
                  ' -i ~/llc_4320/extract/dan_v0/offsets.idx -O ' fout ' ' ...
                  pin(t).name(3:12) ' ' fnm{1} startPoint extent];
        writelines(new_line, 'extract.sh', 'WriteMode', 'append');        
    end
end
% }}}

% {{{ get and save vector 3D fields

% note that zonal velocity is U in faces 1/2 and V in faces 4/5
% and meridional velocity is V in faces 1/2 and -U in faces 4/5
%U
xRange=[3*FS+ix(1) length(ix)];
yRange=[jx(1) length(jx)];
zRange=[1 length(kx)];
maskFile=[maskDir 'hFacS.bits'];
eval(['mkdir ' pout 'U'])
eval(['cd ' pout 'U'])
for t=1:length(tme)
    fin=[pin(t).folder '/V' pin(t).name(2:end)];
    dy=datestr(tme(t),30);
    fout=['U' suf2 '.' dy];
    fld=llc_shrunk_read(fin,maskFile,FS,'xRange',xRange, ...
        'yRange',yRange,'zRange',zRange,'offsets',off_S);
    writebin(fout,fld);
end
%V
xRange=[3*FS+ix(1) length(ix)];
yRange=[jx(1)-1 length(jx)];
zRange=[1 length(kx)];
maskFile=[maskDir 'hFacW.bits'];
eval(['mkdir ' pout 'V'])
eval(['cd ' pout 'V'])
for t=1:length(tme)
    fin=[pin(t).folder '/U' pin(t).name(2:end)];
    dy=datestr(tme(t),30);
    fout=['V' suf2 '.' dy];
    fld=llc_shrunk_read(fin,maskFile,FS,'xRange',xRange, ...
        'yRange',yRange,'zRange',zRange,'offsets',off_W);
    writebin(fout,-fld);
end

% }}}

% {{{ generate commands to extract vector 3D fields
% note that zonal velocity is U in faces 1/2 and V in faces 4/5
% and meridional velocity is V in faces 1/2 and -U in faces 4/5
%U
startPoint=[' ' int2str(3*FS+ix(1)) ',' int2str(jx(1)) ',1 '];
extent=[int2str(length(ix)) ',' int2str(length(jx)) ',' int2str(length(kx))];
eval(['mkdir ' pout 'U'])
eval(['cd ' pout 'U'])
for t=1:length(tme)
    dy=datestr(tme(t),30);
    fout=['U' suf2 '.' dy];
    new_line=['~/llc_4320/extract/dan_v0/extract4320 -d ' pin(t).folder ' -m ' ...
              '/nobackupp29/bcnelson/MITgcm/compressedOutput/work/bitmask.4320x173' ...
              ' -i ~/llc_4320/extract/dan_v0/offsets.idx -O ' fout ' ' ...
              pin(t).name(3:12) ' V' startPoint extent];
    writelines(new_line, 'extract.sh', 'WriteMode', 'append');        
end
%V
startPoint=[' ' int2str(3*FS+ix(1)) ',' int2str(jx(1)-1) ',1 '];
extent=[int2str(length(ix)) ',' int2str(length(jx)) ',' int2str(length(kx))];
eval(['mkdir ' pout 'V'])
eval(['cd ' pout 'V'])
for t=1:length(tme)
    dy=datestr(tme(t),30);
    fout=['V' suf2 '.' dy];
    new_line=['~/llc_4320/extract/dan_v0/extract4320 -d ' pin(t).folder ' -m ' ...
              '/nobackupp29/bcnelson/MITgcm/compressedOutput/work/bitmask.4320x173' ...
              ' -i ~/llc_4320/extract/dan_v0/offsets.idx -n -O ' fout ' ' ...
              pin(t).name(3:12) ' U' startPoint extent];
    writelines(new_line, 'extract.sh', 'WriteMode', 'append');        
end
% }}}

% }}}
